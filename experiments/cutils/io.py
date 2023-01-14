import socket
import time


def on_cluster():
    # hostname = socket.gethostname()
    # return False if hostname == 'coldingham' else True
    return False


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time


def get_project_root():
    if on_cluster():
        path = '/home/s1638128/deployment/decomposition-flows'
    else:
        # path = '/home/conor/Dropbox/phd/projects/decomposition-flows'
        path = 'D:/Academics/Graduate/2021Summer/NSF'
    return path


def get_log_root():
    if on_cluster():
        path = '/home/s1638128/tmp/decomposition-flows/log'
    else:
        path = 'results/'
    return path


def get_data_root():
    if on_cluster():
        path = '/home/s1638128/deployment/decomposition-flows/datasets'
    else:
        path = '/home/conor/Dropbox/phd/projects/decomposition-flows/datasets'
    return path


def get_checkpoint_root(from_cluster=False):
    if on_cluster():
        path = '/home/s1638128/tmp/decomposition-flows/checkpoints'
    else:
        if from_cluster:
            path = '/home/conor/Dropbox/phd/projects/decomposition-flows/checkpoints/cluster'
        else:
            # path = '/home/conor/Dropbox/phd/projects/decomposition-flows/checkpoints'
            path = 'checkpoints/'
    return path


def get_output_root():
    if on_cluster():
        path = '/home/s1638128/tmp/decomposition-flows/out'
    else:
        # path = '/home/conor/Dropbox/phd/projects/decomposition-flows/out'
        path = 'out/'
    return path


def get_final_root():
    if on_cluster():
        path = '/home/s1638128/deployment/decomposition-flows/final'
    else:
        path = 'final/'
    return path


def main():
    print(get_timestamp())


if __name__ == '__main__':
    main()
