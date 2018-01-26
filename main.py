from Configure import configure
import models
import argparse

def _start_shell(local_ns=None):
    ''''''
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

if __name__ == '__main__':

    param = argparse.ArgumentParser(usage='param', description='for main')
    param.add_argument('--model', type=str, default='Single', help='choose one model')
    param.add_argument('--lr', type=float, default=0.001, help='learning rate')
    param.add_argument('--optimizer', type=str, default='adam', help='you can also choose sgd, adagrad, rmsprop')
    param.add_argument('--shell', type=bool, default=False, help='learning rate')
    temparam = param.parse_args()

    config = configure()
    config.lr = temparam.lr
    config.lr_method = temparam.optimizer

    model_ = getattr(models, temparam.model)

    model = model_(config)
    print 'Present used model %s\n'%model
    model.train()


    if temparam.shell:
        print('\n######################\n open ipython for eval')
        _start_shell(locals())