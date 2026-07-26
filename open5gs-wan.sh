#!/usr/bin/env bash
set -Eeuo pipefail

# Open5GS settings
TUN_IF="${TUN_IF:-ogstun}"
UE_IPV4_SUBNET="${UE_IPV4_SUBNET:-10.45.0.0/16}"
UE_IPV6_SUBNET="${UE_IPV6_SUBNET:-2001:db8:cafe::/48}"
ENABLE_IPV6="${ENABLE_IPV6:-yes}"

# Set to "yes" only when you intentionally want this script to disable UFW.
DISABLE_UFW="${DISABLE_UFW:-no}"

if [[ $EUID -ne 0 ]]; then
    echo "Run this script with sudo:"
    echo "  sudo $0"
    exit 1
fi

# Detect the interface used for the default Internet route.
WAN_IF="${WAN_IF:-$(ip -4 route show default |
    awk '/default/ {print $5; exit}')}"

if [[ -z "$WAN_IF" ]]; then
    echo "ERROR: Could not detect the WAN interface."
    echo "Set it manually, for example:"
    echo "  sudo WAN_IF=eno1 $0"
    exit 1
fi

if ! ip link show "$TUN_IF" >/dev/null 2>&1; then
    echo "ERROR: Open5GS tunnel interface '$TUN_IF' does not exist."
    echo "Check it with: ip addr show $TUN_IF"
    exit 1
fi

echo "Open5GS tunnel : $TUN_IF"
echo "WAN interface   : $WAN_IF"
echo "IPv4 UE subnet : $UE_IPV4_SUBNET"
echo "IPv6 UE subnet : $UE_IPV6_SUBNET"

# ----------------------------------------------------------------------
# 1. Enable forwarding permanently
# ----------------------------------------------------------------------

cat > /etc/sysctl.d/99-open5gs-wan.conf <<SYSCTL
net.ipv4.ip_forward=1
net.ipv6.conf.all.forwarding=1
SYSCTL

sysctl -w net.ipv4.ip_forward=1

if [[ "$ENABLE_IPV6" == "yes" ]]; then
    sysctl -w net.ipv6.conf.all.forwarding=1
fi

# ----------------------------------------------------------------------
# Helper functions: add rules only when they do not already exist
# ----------------------------------------------------------------------

add_iptables_rule() {
    local table="$1"
    shift

    if ! iptables -t "$table" -C "$@" 2>/dev/null; then
        iptables -t "$table" -A "$@"
    fi
}

add_ip6tables_rule() {
    local table="$1"
    shift

    if ! ip6tables -t "$table" -C "$@" 2>/dev/null; then
        ip6tables -t "$table" -A "$@"
    fi
}

# ----------------------------------------------------------------------
# 2. IPv4 NAT and forwarding
# ----------------------------------------------------------------------

add_iptables_rule nat POSTROUTING \
    -s "$UE_IPV4_SUBNET" ! -o "$TUN_IF" -j MASQUERADE

# Permit UE-originated traffic toward the WAN.
add_iptables_rule filter FORWARD \
    -i "$TUN_IF" -o "$WAN_IF" -s "$UE_IPV4_SUBNET" -j ACCEPT

# Permit response traffic back to the UE.
add_iptables_rule filter FORWARD \
    -i "$WAN_IF" -o "$TUN_IF" -d "$UE_IPV4_SUBNET" \
    -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

# ----------------------------------------------------------------------
# 3. IPv6 NAT and forwarding
# ----------------------------------------------------------------------

if [[ "$ENABLE_IPV6" == "yes" ]]; then
    if ip6tables -t nat -L >/dev/null 2>&1; then
        add_ip6tables_rule nat POSTROUTING \
            -s "$UE_IPV6_SUBNET" ! -o "$TUN_IF" -j MASQUERADE

        add_ip6tables_rule filter FORWARD \
            -i "$TUN_IF" -o "$WAN_IF" -s "$UE_IPV6_SUBNET" -j ACCEPT

        add_ip6tables_rule filter FORWARD \
            -i "$WAN_IF" -o "$TUN_IF" -d "$UE_IPV6_SUBNET" \
            -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
    else
        echo "WARNING: IPv6 NAT table is unavailable; skipping IPv6 NAT."
    fi
fi

# ----------------------------------------------------------------------
# 4. UFW handling
# ----------------------------------------------------------------------

if command -v ufw >/dev/null 2>&1 &&
   ufw status 2>/dev/null | grep -q "Status: active"; then

    if [[ "$DISABLE_UFW" == "yes" ]]; then
        echo "Disabling UFW as requested..."
        ufw disable
    else
        echo
        echo "WARNING: UFW is active and may block forwarded UE traffic."
        echo "To disable it using this script, run:"
        echo "  sudo DISABLE_UFW=yes $0"
    fi
fi

echo
echo "Open5GS WAN routing configuration completed."
echo
echo "IPv4 forwarding:"
sysctl net.ipv4.ip_forward

echo
echo "IPv4 NAT rule:"
iptables -t nat -S POSTROUTING |
    grep -F "$UE_IPV4_SUBNET" || true

echo
echo "Forwarding rules:"
iptables -S FORWARD |
    grep -E "$TUN_IF|$UE_IPV4_SUBNET" || true
