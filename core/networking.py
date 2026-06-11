import socket
import psutil
import logging

logger = logging.getLogger("Core.Networking")

def get_network_info():
    """
    Retrieves local network interfaces (IPv4/IPv6) and determines the preferred LAN IP.
    Returns:
        dict: {
            "ipv4": [str],
            "ipv6": [str],
            "preferred": str
        }
    """
    ipv4s = []
    ipv6s = []
    preferred_ip = "127.0.0.1"

    try:
        # Iterate over all interfaces
        for interface, snics in psutil.net_if_addrs().items():
            for snic in snics:
                if snic.family == socket.AF_INET:
                    ip = snic.address
                    if ip == "127.0.0.1":
                        continue
                    ipv4s.append(ip)
                    # Heuristic: 192.168.x.x or 10.x.x.x or 172.16-31.x.x are usually LAN
                    if ip.startswith("192.168.") or ip.startswith("10.") or (ip.startswith("172.") and 16 <= int(ip.split('.')[1]) <= 31):
                        preferred_ip = ip
                
                elif snic.family == socket.AF_INET6:
                    ip = snic.address.split('%')[0] # Remove scope ID if present
                    if ip != "::1" and not ip.startswith("fe80"): # Skip link-local if desired, but maybe keep for reference
                        ipv6s.append(ip)

        # Try to find the actual route to the internet for the "True" preferred IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # We don't actually connect, just determine the route
                s.connect(("8.8.8.8", 80))
                route_ip = s.getsockname()[0]
                if route_ip in ipv4s:
                    preferred_ip = route_ip
        except Exception:
            pass # Fallback to heuristic or loopback

    except Exception as e:
        logger.error(f"Error retrieving network info: {e}")

    return {
        "ipv4": sorted(list(set(ipv4s))),
        "ipv6": sorted(list(set(ipv6s))),
        "preferred": preferred_ip
    }
