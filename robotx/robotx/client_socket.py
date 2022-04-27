import socket

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 1010)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

#construct message
message = [1,3]
message=bytearray(message)

#send message
sock.sendall(message)

print('closing socket')
sock.close()