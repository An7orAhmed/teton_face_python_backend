# bash iptest.sh

ip route show
sudo ip route del default via 192.168.1.10 dev eth0
sudo ip route add default via 10.0.0.1 dev wlan0
sudo ip route add 192.168.1.0/24 dev eth0
ip route show