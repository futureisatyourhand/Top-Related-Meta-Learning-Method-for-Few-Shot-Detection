from darknet_meta_fpn import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os
import pdb


def valid(datacfg, darknetcfg, learnetcfg,fpncfg, weightfile, outfile, use_baserw=False):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    metadict = options['meta']

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(darknetcfg, learnetcfg,fpncfg,"ours")
    m.print_network()
    m.load_weights(weightfile,train=False)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 


    if True:
        metaset = dataset.MetaDataset(metafiles=metadict, train=False, ensemble=True,with_ids=True)
        metaloader = torch.utils.data.DataLoader(
            metaset,
            batch_size=len(metaset),
            shuffle=False,
            **kwargs
        )
        #metaloader = iter(metaloader)
        n_cls = len(metaset.classes)
        print(metaset.metalines)
        dwall=[[]]*n_cls
        for metax, mask, clsids in metaloader:
             metax, mask = metax.cuda(), mask.cuda() 
             metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
             dws = m.meta_forward(metax, mask)
             dw=dws[0]
             print(dw.shape,metax.shape,len(metaloader))
             for ci, c in enumerate(clsids):
                    #enews[c] = enews[c] * cnt[c] / (cnt[c] + 1) + dw[ci] / (cnt[c] + 1)
                    #cnt[c] += 1
                    dwall[c].append(dw[ci])
        dwall=[torch.stack(d) for d in dwall]
        dwall=torch.stack(dwall)
        import pickle
        print('====all classes===',cfg.classes)
        print('====novel classes===',cfg.novel_ids)
        print('====base classes===',cfg.base_ids)
        with open('data/rws/voc_novel2_all.pkl','wb') as f:
             tmp=dwall.data.cpu().numpy()
             pickle.dump(tmp,f)
        print('===> Generating dynamic weights...')
        #metax, mask = metaloader.next()
        #metax, mask = metax.cuda(), mask.cuda()
        #metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
        #dynamic_weights = m.meta_forward(metax, mask)

        #for i in range(len(dynamic_weights)):
        #    assert dynamic_weights[i].size(0) == sum(metaset.meta_cnts)
        #    inds = np.cumsum([0] + metaset.meta_cnts)
        #    new_weight = []
        #    for j in range(len(metaset.meta_cnts)):
        #        new_weight.append(torch.mean(dynamic_weights[i][inds[j]:inds[j+1]], dim=0))
        #    dynamic_weights[i] = torch.stack(new_weight)
        #    print(dynamic_weights[i].shape)
    else:
        metaset = dataset.MetaDataset(metafiles=metadict, train=False, ensemble=True, with_ids=True)
        metaloader = torch.utils.data.DataLoader(
            metaset,
            batch_size=64,
            shuffle=False,
            **kwargs
        )
        # metaloader = iter(metaloader)
        n_cls = len(metaset.classes)

        enews = [0.0] * n_cls
        cnt = [0.0] * n_cls
        dwall=[[]]*n_cls
        print('===> Generating dynamic weights...')
        print("===all classes===",cfg.classes)
        print('====base ids====',cfg.base_ids)
        print('=====novel ids====',cfg.novel_ids)
        print(metaset.metalines)
        kkk = 0
        for metax, mask, clsids in metaloader:
            print('===> {}/{}'.format(kkk, len(metaset) // 64),clsids,metax.shape,mask.shape)
            kkk += 1
            metax, mask = metax.cuda(), mask.cuda()
            
            metax, mask = Variable(metax, volatile=True), Variable(mask, volatile=True)
            dws = m.meta_forward(metax, mask)
            dw = dws[0]
            for ci, c in enumerate(clsids):
                enews[c] = enews[c] * cnt[c] / (cnt[c] + 1) + dw[ci] / (cnt[c] + 1)
                cnt[c] += 1
                dwall[c].append(dw[ci])
        dwall=[torch.stack(d) for d in dwall]
        dd=torch.stack(dwall)
        dynamic_weights = [torch.stack(enews)]

        import pickle
        with open('data/rws/voc_novel2_.pkl', 'wb') as f:
            tmp = [x.data.cpu().numpy() for x in dynamic_weights]
            pickle.dump(tmp, f)
        with open('data/rws/voc_novel2_all.pkl','wb') as f:
            tmp=dd.data.cpu().numpy()
            pickle.dump(tmp,f)
        # import pdb; pdb.set_trace()

        if use_baserw:
            import pickle
            # f = 'data/rws/voc_novel{}_.pkl'.format(cfg.novelid)
            f = 'data/rws/voc_novel{}_.pkl'.format(0)
            print('===> Loading from {}...'.format(f))
            with open(f, 'rb') as f:
            # with open('data/rws/voc_novel0_.pkl', 'rb') as f:
                rws = pickle.load(f)
                rws = [Variable(torch.from_numpy(rw)).cuda() for rw in rws]
                tki = cfg._real_base_ids
                for i in range(len(rws)):
                    dynamic_weights[i][tki] = rws[i][tki]
                    # dynamic_weights[i] = rws[i]
            # pdb.set_trace()


        #fps_ious[i].close()

    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    import sys
    if len(sys.argv) in [5,6,7]:
        datacfg = sys.argv[1]
        darknet = parse_cfg(sys.argv[2])
        learnet = parse_cfg(sys.argv[3])
        fpn=parse_cfg(sys.argv[4])
        weight = sys.argv[5]
        if len(sys.argv) >= 6:
            gpu = sys.argv[6]
        else:
            gpu = '0'
        if len(sys.argv) == 8:
            use_baserw = True
        else:
            use_baserw = False

        data_options  = read_data_cfg(datacfg)
        net_options   = darknet[0]
        meta_options  = learnet[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # Configure options
        cfg.config_data(data_options)
        cfg.config_meta(meta_options)
        cfg.config_net(net_options)

        outfile = 'comp4_det_test_'
        #for w in os.listdir(weight):
        #    weightfile=weight+w
            #if w[:4] == "0002":# or ("0003" in weightfile):
        valid(datacfg, darknet, learnet,fpn, weight, outfile, use_baserw)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
