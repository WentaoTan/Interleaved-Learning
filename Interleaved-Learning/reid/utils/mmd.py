import torch
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    ��Դ�����ݺ�Ŀ��������ת��Ϊ�˾��󣬼������е�K
    Params:
	    source: Դ�����ݣ�n * len(x))
	    target: Ŀ�������ݣ�m * len(y))
	    kernel_mul:
	    kernel_num: ȡ��ͬ��˹�˵�����
	    fix_sigma: ��ͬ��˹�˵�sigmaֵ
	Return:
		sum(kernel_val): ����˾���֮��
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# ������������һ��source��target�ĳ߶���һ���ģ��������ڼ���
    total = torch.cat([source, target], dim=0)#��source,target���з���ϲ�
    #��total���ƣ�n+m����
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #��total��ÿһ�ж����Ƴɣ�n+m���У���ÿ�����ݶ���չ�ɣ�n+m����
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #��������������֮��ĺͣ��õ��ľ��������꣨i,j������total�е�i�����ݺ͵�j������֮���l2 distance(i==jʱΪ0��
    L2_distance = ((total0-total1)**2).sum(2)
    #������˹�˺�����sigmaֵ
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #��fix_sigmaΪ��ֵ����kernel_mulΪ����ȡkernel_num��bandwidthֵ������fix_sigmaΪ1ʱ���õ�[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #��˹�˺�������ѧ���ʽ
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #�õ����յĺ˾���
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    ����Դ�����ݺ�Ŀ�������ݵ�MMD����
    Params:
	    source: Դ�����ݣ�n * len(x))
	    target: Ŀ�������ݣ�m * len(y))
	    kernel_mul:
	    kernel_num: ȡ��ͬ��˹�˵�����
	    fix_sigma: ��ͬ��˹�˵�sigmaֵ
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#һ��Ĭ��ΪԴ���Ŀ�����batchsize��ͬ
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #����ʽ��3�����˾���ֳ�4����
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#��Ϊһ�㶼��n==m������L����һ�㲻�������