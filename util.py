import socket
import pickle
import time


host = '127.0.0.1'
port = 54000

def get_last_line(file_name):
    '''
    Open file and read the last line

    Args:
        file_name (str): file name

    Returns:
        str: last line of the file, '' if the file not found
    '''
    try:
        with open(file_name, 'rb') as file:
            file.seek(-2,2)
            while file.read(1)!=b'\n':
                file.seek(-2, 1)
            return file.readline().decode().strip()
    except FileNotFoundError:
        return ''

def get_pool_stats():
    '''
    Communicating with cache server through socket, get current cache pool name and cache allocation

    Args:

    Returns:
        map<poolname->poolsize>: all pool in the cache and their pool size
    '''
    # link the target program
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    message = "G:"
    # print(message)
    sock.sendall(message.encode())

    # wait the response
    response = sock.recv(1024).decode()
    # print(response)
    sock.close()
    deserialized_map = {}
    pairs = response.split(';')[:-1]
    for pair in pairs:
        key,value = pair.split(':')
        deserialized_map[key] = int(value)
    
    return deserialized_map

def set_cache_size(workloads, cache_size):
    '''
    Communicating with cache server through socket, adjust pool size to the new size

    Args:
        workloads (list<str>): pool name of all workloads
        cache_size (list<int>): new size of each pool
    
    Returns:
    '''
    # link the server program
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    curr_config = [workloads, cache_size]
    serialized_data = '\n'.join([' '.join(map(str, row)) for row in curr_config])
    serialized_data = 'S:' + serialized_data
    #print(serialized_data)

    # send to server
    sock.sendall(serialized_data.encode())
    sock.close()

def receive_config():
    '''
    Old version, Wait to receive the current resource config

    Args:

    Returns:
        list: [
            [name of pool], 
            [allocation of resource A of every pool], # such as cache size,[16,16]
            [context of resource A of every pool ], # for cache size ,context canbe hit_rate,[0.8254, 0.7563] 
            [latency of every warkload]
        ]
    '''
    # create the server socket
    # print("receive config")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 1412))
    server_socket.listen(1)
    
    # listening the old_config from the executor
    client_socket, _ = server_socket.accept()
    received_data = client_socket.recv(1024)
    config = pickle.loads(received_data)
    
    client_socket.close()
    server_socket.close()
    return config

def send_config(new_config):
    '''
    Old version, Send the new config

    Args:
        list: [
            [name of pool],
            [new allocation of resource A of every pool]
        ]

    Returns:
    '''
    serialized_config = pickle.dumps(new_config)

    # connect to the bench
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 1413))
    client_socket.send(serialized_config)
    client_socket.close()
    print("send_config success!")
    return

class config_management:
    def __init__(self) -> None:
        pass

    def receive_config(self):
        curr_config = []
        # pool name and cache allocation
        pool_and_size = get_pool_stats()
        pool_name = list(pool_and_size.keys())
        pool_size = list(pool_and_size.values())
        curr_config.append(pool_name)
        curr_config.append(pool_size)
        #TODO: cache hit rate
        hitrate_log = ['/home/md/SHMCachelib/bin/' + name + '_hitrate.log' for name in pool_name]
        hitrates = []
        for log in hitrate_log:
            hitrates.append(get_last_line(log))
        curr_config.append(hitrates)
        #TODO: tail latency
        latency_log = ['/home/md/SHMCachelib/bin/' + name + '_tailLatency.log' for name in pool_name]
        latencies = []
        for log in latency_log:
            latencies.append(get_last_line(log))
        curr_config.append(latencies)

        return curr_config

    def send_config(self, new_config):
        set_cache_size(new_config[0], new_config[1])

def userA_func(resources, threshold):
    if threshold < 0:
        return 0
    if resources < threshold:
        return resources * 0.095
    else:
        return threshold * 0.095 + (resources - threshold) * 0.010

def userB_func(resources, threshold):
    if threshold < 0:
        return 0
    if resources < threshold:
        return resources * 0.040
    else:
        return threshold * 0.040 + (resources - threshold) * 0.005

# def userC_func(resources, threshold):
#     if threshold < 0:
#         return 0
#     if resources < threshold:
#         return resources * 0.040
#     else:
#         return threshold * 0.040 + (resources - threshold) * 0.005

class simulation_config_management:
    def __init__(self):
        self.total_resource = 32
        self.userA = self.total_resource / 2
        self.userB = self.total_resource / 2
        self.counter = 0

    def receive_config(self):
        workload_change = 401
        self.counter = self.counter + 1
        if self.counter == workload_change:
            print('---------------------------workload change -----------------------------')
        curr_config = []
        curr_config.append(['userA', 'userB'])
        curr_config.append([self.userA, self.userB])
        if self.counter < workload_change:
            curr_config.append([userA_func(self.userA, 8), userB_func(self.userB, 24)])
        else:
            curr_config.append([userB_func(self.userA, 24), userA_func(self.userB, 8)])
        curr_config.append(self.total_resource)
        return curr_config
    
    def send_config(self, new_config):
        self.userA = new_config[1][0]
        self.userB = new_config[1][1]


if __name__ == '__main__':
    time.sleep(10)
    cs = config_management()
    curr = cs.receive_config()
    for item in curr:
        print(item)