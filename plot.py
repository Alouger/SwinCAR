import matplotlib.pyplot as plt
import os
def train_loss_plot(epoch,Model,BATCH_SIZE,dataset,loss):
    num = epoch
    Model = Model
    batch_size = BATCH_SIZE
    dataset = dataset
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(Model)+'_'+str(batch_size)+'_'+str(dataset)+'_'+str(epoch)+'_train_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)
    
def val_loss_plot(epoch,Model,BATCH_SIZE,dataset,loss):
    num = epoch
    Model = Model
    batch_size = BATCH_SIZE
    dataset = dataset
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(Model)+'_'+str(batch_size)+'_'+str(dataset)+'_'+str(epoch)+'_val_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

def metrics_plot(epoch,Model,BATCH_SIZE,dataset,name,*args):
    num = epoch
    Model = Model
    batch_size = BATCH_SIZE
    dataset = dataset
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(Model) + '_' + str(batch_size) + '_' + str(dataset) + '_' + str(epoch) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)
    
def test_metrics_plot(epoch,Model,BATCH_SIZE,dataset,name,*args):
    num = epoch
    Model = Model
    batch_size = BATCH_SIZE
    dataset = dataset
    names = name.split('&')
    metrics_value = args
    i=0
    # for i in range(num)

    x = [i for i in range(num)]
    plot_save_path = r'result/temp_out'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(Model) + '_' + str(batch_size) + '_' + str(dataset) + '_' + str(epoch) + '_'+name+'.jpg'
    plt.figure()
    print(metrics_value)
    # for l in metrics_value:
    plt.plot(x,metrics_value,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        # i+=1
    plt.legend()
    plt.savefig(save_metrics)