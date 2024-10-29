import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_selection import f_regression, mutual_info_regression, mutual_info_classif
from sklearn import metrics
import scipy.optimize as opt
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import argparse
def discrete_mi_est(xs, ys, nx=2, ny=2):
    prob = np.zeros((nx, ny))
    for a, b in zip(xs, ys):
        prob[a,b] += 1.0/len(xs)
    pa = np.sum(prob, axis=1)
    pb = np.sum(prob, axis=0)
    mi = 0
    chi = 0
    hd = 0
    jsd = 0
    # zcp = 0
    tv = 0
    for a in range(nx):
        for b in range(ny):
            if prob[a,b] < 1e-9:
                continue
            dQ = pa[a]*pb[b]
            dP = prob[a,b]
            dR = dP / dQ
            mi += dP * np.log(dR)
            chi += dQ * np.square(dR-1)
            hd += dQ * np.square(np.sqrt(dR)-1)
            jsd += dQ * (dR * np.log(2*dR/(1+dR)) + np.log(2/(1+dR)))
            # zcp += dQ * np.abs(dR-1) * np.sqrt(np.log(1+np.square(dR-1)))
            tv += dQ * np.abs(dR-1)

    mi = max(0.0, mi)
    chi = max(0.0, chi)
    hd = max(0.0, hd)
    jsd = max(0.0, jsd)
    # zcp = max(0.0, zcp)
    tv = max(0.0, tv)
    return mi, chi, hd, jsd, tv

def SuperSample(num=2, num_class=2, d=2, sep=1.0):
  samples = make_classification(n_samples=2*num, n_features=d, n_redundant=0, n_classes=num_class, n_informative=d, n_clusters_per_class=1, class_sep=sep, flip_y=0)
  red = samples[0][samples[1] == 0]
  blue = samples[0][samples[1] == 1]
  for i in range(int(num_class)):
    cluster = samples[0][samples[1] == i]
    targets = np.zeros(len(cluster))+i
    if i==0:
      inputs = cluster
      labels = targets
    else:
      labels = np.append(labels,targets)
      inputs = np.concatenate((inputs,cluster),axis=0)

  X_0, X_1, y_0,  y_1 = train_test_split(
      inputs, labels, test_size=0.5, random_state=42)
  return X_0, X_1, y_0, y_1


def TrainClassification(X_0, X_1, y_0, y_1, U, d, num_class):
  class MultiClassification(torch.nn.Module):
      def __init__(self, input_dim, output_dim):
          super(MultiClassification, self).__init__()
          self.linear = torch.nn.Linear(input_dim, output_dim)

      def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


  criterion = torch.nn.CrossEntropyLoss()
  epochs = 300
  input_dim = d
  output_dim = num_class
  learning_rate = 0.01
  model = MultiClassification(input_dim,output_dim)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  X_train =(1-U).reshape(-1, 1)*X_0+U.reshape(-1, 1)*X_1
  y_train = (1-U)*y_0+U*y_1

  X_train, y_train = torch.Tensor(X_train),torch.Tensor(y_train).type(torch.LongTensor)
  X_0, y_0, X_1, y_1 = torch.Tensor(X_0),torch.Tensor(y_0).type(torch.LongTensor), torch.Tensor(X_1),torch.Tensor(y_1).type(torch.LongTensor)

  losses = []
  losses_test = []
  Iterations = []
  iter = 0
  for epoch in range(int(epochs)):
      x = X_train
      labels = y_train
      optimizer.zero_grad()
      outputs = model(X_train)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        total = 0
        correct = 0
        total += y_train.size(0)
        correct += np.sum(outputs.data.max(1)[1].detach().view_as(y_train).numpy() == y_train.detach().numpy())
        accuracy = 100 * correct/total
        if accuracy>=99:
          break

  with torch.no_grad():
      outputs_0 = torch.squeeze(model(X_0))
      predicted_0 = outputs_0.data.max(1)[1].detach().view_as(y_0).numpy()
      Err0= (predicted_0 != y_0.detach().numpy()).astype(int)

      outputs_1 = torch.squeeze(model(X_1))
      predicted_1 = outputs_1.data.max(1)[1].detach().view_as(y_1).numpy()
      Err1= (predicted_1 != y_1.detach().numpy()).astype(int)
      Delta = Err1-Err0

  return Err0, Err1, Delta, predicted_0.astype(int), predicted_1.astype(int)


def RunClassExp(n=2, n_class=2, dim=2, diff=1.0, num_Z=50, num_U=100):
  disint_mi, disint_chi, disint_hd, disint_jsd, disint_tv=0, 0, 0, 0, 0
  mi, chi, hd, jsd=0, 0, 0, 0
  disint_mi1, disint_mi2, disint_mi3=0, 0, 0
  disint_hd1, disint_hd2 = 0, 0
  perins_Z=0
  for i in range(int(num_Z)):
    X_0, X_1, y_0, y_1 = SuperSample(num=n, num_class=n_class, d=dim, sep=diff)
    first_train, second_train = 0, 0
    first_trerr, second_trerr = 0, 0
    for j in range(int(num_U)):
      U=np.random.binomial(size=n, n=1, p=0.5)
      Err0, Err1, Delta, Yhat0, Yhat1 = TrainClassification(X_0, X_1, y_0, y_1, U, dim, n_class)
      error = ((-1)**U*Delta).mean()
      first_train += 1-U
      second_train += U
      first_trerr += (1-U)*Err0
      second_trerr += U*Err1
      train_error = ((1-U)*Err0+U*Err1).mean()
      tr_square = train_error**2

      if j==0 and i==0:
        L0, L1, Del, Err, Tr_err, Trerr_square=Err0, Err1, Delta, error, train_error, tr_square
        F0, F1, Rad = Yhat0, Yhat1, U #if i==0 else np.vstack((Rad,U))
      else:
        L0, L1, Del=np.vstack((L0,Err0)), np.vstack((L1,Err1)), np.vstack((Del,Delta))
        Err, Tr_err, Trerr_square=np.append(Err,error), np.append(Tr_err,train_error), np.append(Trerr_square,tr_square)
        F0, F1, Rad=np.vstack((F0,Yhat0)), np.vstack((F1,Yhat1)), np.vstack((Rad,U))


    perins_Z += (first_trerr/first_train)**2+(second_trerr/second_train)**2
    # midelta, fcmi = np.zeros(n), np.zeros(n)
    midelta, ld_square, ld_mean = np.zeros(n), np.zeros(n), np.zeros(n)
    chidelta, hddelta, jsddelta, tvdelta = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
      Delvec = Del[i*num_U:(i+1)*num_U-1,k]
      ld_square[k]=np.square(Delvec).mean()
      ld_mean[k]=np.abs(np.mean(Delvec))
      # print(Rad[i*num_U:(i+1)*num_U-1,k])
      # print((Rad[i*num_U:(i+1)*num_U-1,k]).tolist)
      # print()
      # print((np.array(Delvec)+1).tolist)
      midelta[k], chidelta[k], hddelta[k], jsddelta[k], tvdelta[k] =discrete_mi_est((Rad[i*num_U:(i+1)*num_U-1,k]), (np.array(Delvec)+1), nx=2, ny=3)
      # midelta[k]=mutual_info_classif(Del[i*num_U:(i+1)*num_U-1,k].reshape(-1, 1), Rad[i*num_U:(i+1)*num_U-1,k], discrete_features=[True])
      # fcmi[k]=mutual_info_classif((n_class*F0[i*num_U:(i+1)*num_U-1,k]+F1[i*num_U:(i+1)*num_U-1,k]).reshape(-1, 1), Rad[i*num_U:(i+1)*num_U-1,k], discrete_features=[True])

    disint_mi += (np.sqrt(2*midelta)).mean()
    disint_mi1 += (np.sqrt(2* (ld_square + ld_mean) * midelta)).mean()
    disint_mi2 += (midelta + np.sqrt((midelta + 2 * ld_square) * midelta)).mean()
    disint_mi3 += (np.sqrt(2 * (ld_square + tvdelta) * midelta)).mean()
    disint_hd += (np.sqrt((4 * ld_square + 2* ld_mean) * hddelta)).mean()
    disint_hd1 += (np.sqrt((4 * ld_square + 2* tvdelta) * hddelta)).mean()
    disint_hd2 += (np.sqrt((4 + 2* tvdelta) * hddelta)).mean()
    disint_jsd += (2 * np.sqrt((4 * ld_square + ld_mean) * jsddelta)).mean()
    disint_chi += (np.sqrt(2*(ld_square + ld_mean) * chidelta)).mean()

    mi += (np.sqrt(midelta)).mean()
    hd += (np.sqrt(hddelta)).mean()
    jsd += (np.sqrt(jsddelta)).mean()
    chi += (np.sqrt(chidelta)).mean()


  # mi, mil0, mil1, mipair=np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
  mil0, mipair=np.zeros(n),np.zeros(n)
  for i in range(n):
  #   mi[i]=mutual_info_classif(Del[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
    mil0[i]=mutual_info_classif(L0[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
  #   mil1[i]=mutual_info_classif(L1[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
    mipair[i]=mutual_info_classif((n_class*L0[:,i]+L1[:,i]).reshape(-1, 1), Rad[:,i], discrete_features=[True])

  # bound=(np.sqrt(2*mi)).mean()
  # bound0=2*(np.sqrt(2*mil0)).mean()
  # bound1=2*(np.sqrt(2*mil1)).mean()
  # bound2=(np.sqrt(2*mipair)).mean()
  disintbound=disint_mi/num_Z
  mibound1=disint_mi1/num_Z
  mibound2=disint_mi2/num_Z
  mibound3=disint_mi3/num_Z
  hdbound=disint_hd/num_Z
  hdbound1=disint_hd1/num_Z
  hdbound2=disint_hd2/num_Z
  jsdbound=disint_jsd/num_Z
  chibound=disint_chi/num_Z

  mi_avg=mi/num_Z
  hd_avg=hd/num_Z
  jsd_avg=jsd/num_Z
  chi_avg=chi/num_Z
  # fcmibound=disint_fcmi/num_Z
  # bound3=(np.sqrt(2*fcmi)).mean()
  # inpbound1= 2*mil0.mean()/np.log(2)
  # inpbound2= mipair.mean()/np.log(2)
  L_n = Tr_err.mean()
  I_mean = mil0.mean()
  forsharp=perins_Z/(num_Z*2)
  # subbound=4*I_mean+4*np.sqrt(I_mean*L_n)
  # exactbound=mi.mean()/np.log(2)

  # fun1 = lambda x: x[0]*L_n + I_mean/x[1]
  # cons1 = ({'type': 'ineq', 'fun': lambda x:  -np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]+1)) + 2})
  # bnds1 = ((0, None), (0, None))

  # # init = 2*np.sqrt(I_mean/L_n)
  # # (0.5, 0.1)
  # res1 = opt.minimize(fun1, (0.5, 0.1), method='SLSQP', bounds=bnds1,
  #              constraints=cons1, options = {'disp':False})
  # optbound = res1.fun

  # fun2 = lambda x: x[0]*(L_n-(1-x[2]**2)*Trerr_square.mean()) + I_mean/x[1]
  cons2 = ({'type': 'ineq', 'fun': lambda x:  (-np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]*(x[2]**2)+1)) + 2)*x[1]})
  bnds2 = ((0, None), (0, None), (0, 1))

  # res2 = opt.minimize(fun2, (1, 0.1, 0.5), method='SLSQP', bounds=bnds2,
  #              constraints=cons2, options = {'disp':False})
  # varbound = res2.fun

  fun3 = lambda x: x[0]*(L_n-(1-x[2]**2)*forsharp.mean()) + I_mean/x[1]

  res3 = opt.minimize(fun3, (1, 0.1, 0.5), method='SLSQP', bounds=bnds2,
               constraints=cons2, options = {'disp':False})
  sharpbound = res3.fun

  def kl(q,p):
    if q>0:
        return q*np.log(q/p) + (1-q)*np.log( (1-q)/(1-p) )
    else:
        return np.log( 1/(1-p) )
  # def conkl(R):
  #   return (mipair.mean()-kl(L_n,L_n/2 + R/2))*R*(1-R)
  conkl = ({'type': 'ineq', 'fun': lambda x:  mipair.mean()-kl(L_n,L_n/2 + x[0]/2)},
        {'type': 'ineq', 'fun': lambda x: x[0]*(1-x[0])},
       {'type': 'ineq', 'fun': lambda x: 1-L_n/2-x[0]/2})
  objective = lambda R: -R
  # conskl = ({'type': 'ineq', 'fun' : lambda R: (mipair.mean()-kl(L_n,L_n/2 + R/2))*(1-L_n/2-R/2)})
  results = opt.minimize(objective, 0.5, method='SLSQP',
  constraints = conkl,
  options = {'disp':False})
  klbound = results.x[0]
  return disintbound, mibound1, mibound2, mibound3, hdbound, hdbound1, hdbound2, jsdbound, chibound, mi_avg, hd_avg, jsd_avg, chi_avg, sharpbound, klbound, Err.mean(), Tr_err.mean()


import warnings
warnings.filterwarnings("ignore")
def PlotBound(n_class=2, dim=5, diff=1, num_Z=50, num_U=100, sample=[5, 10, 20, 30, 50], exp_name='two-simple'):
  print("------Class:", n_class, " Difficulty:", 1/diff,"------")
  disint_bound, mi_bound1, mi_bound2, mi_bound3, hd_bound, jsd_bound, chi_bound =[], [], [], [], [], [], []
  sharp_bound, kl_bound, mi_avg, hd_avg, jsd_avg, chi_avg = [], [], [], [], [], []
  hd_bound1, hd_bound2 = [], []
  err, tr_loss = [], []
  for num in sample:
    disint, mi1, mi2, mi3, hd, hd1, hd2, jsd, chi, cmi, chdi, cjsdi, cchii, sharpcmi, klcmi, em_err, em_trloss = RunClassExp(n=num, n_class=n_class, dim=dim, diff=diff, num_Z=num_Z, num_U=num_U)

    print('Num %d: LD: %.3f | MI1: %.3f | MI2: %.3f | MI3: %.3f | HD: %.3f |HD1: %.3f |HD2: %.3f | JSD: %.3f | Chi: %.3f '
                          '| SharpFast: %.3f | BiKL: %.3f | Gap: %.3f | Train: %.3f' %
                          (num, disint, mi1, mi2, mi3, hd, hd1, hd2, jsd, chi, sharpcmi, klcmi, em_err, em_trloss))
    print('Num %d: CMI: %.3f | CHDI: %.3f | CJSDI: %.3f | CChiI: %.3f ' %
                          (num, cmi, chdi, cjsdi, cchii))
    disint_bound.append(disint)
    mi_bound1.append(mi1)
    mi_bound2.append(mi2)
    mi_bound3.append(mi3)
    hd_bound.append(hd)
    hd_bound1.append(hd1)
    hd_bound2.append(hd2)
    jsd_bound.append(jsd)
    chi_bound.append(chi)
    sharp_bound.append(sharpcmi)
    kl_bound.append(klcmi)
    mi_avg.append(cmi)
    hd_avg.append(chdi)
    jsd_avg.append(cjsdi)
    chi_avg.append(cchii)
    err.append(np.abs(em_err))
    tr_loss.append(em_trloss)

  # plt.figure()
  tmax = max([disint_bound[0], mi_bound1[0], mi_bound2[0], mi_bound3[0], hd_bound1[0], hd_bound2[0], hd_bound[0], jsd_bound[0], chi_bound[0], sharp_bound[0], kl_bound[0]])
  plt.figure(figsize=(6, 4))
  plt.plot(sample,mi_bound1, color='orange', label="MI1")
  plt.plot(sample,mi_bound2, color='red', label="MI2")
  plt.plot(sample,mi_bound3, color='green', label="MI3")
  plt.plot(sample,hd_bound, color='blue', label="HD")
  plt.plot(sample,hd_bound1, label="HD1")
  plt.plot(sample,hd_bound2, label="HD2")
  plt.plot(sample,disint_bound, color='magenta', label="disInt")
  plt.plot(sample,jsd_bound, color='pink', label="JSD-bound")
  plt.plot(sample,chi_bound, color='cyan', label="Chi^2-bound")
  # plt.plot(sample,pair_ip_bound, color='magenta', label="PIP-bound")
  plt.plot(sample,tr_loss, color='black', label="TrErr")
  # plt.plot(sample,sub_bound, color='yellow', label="Sub")
  # plt.plot(sample,opt_bound, color='pink', label="Opt")
  # plt.plot(sample,var_bound, color='green', label="Var")
  plt.plot(sample,sharp_bound, marker = 'x',markersize = 8, label="Sharp")
  plt.plot(sample,kl_bound, marker = '2',markersize = 8, label="KL")
  plt.plot(sample,err, color='gray', label="Err")
  # plt.plot(sample,exact_bound, label="Perfect-bound")
  plt.legend()
  plt.ylim(-0.05, tmax + 0.02)
  plt.savefig(exp_name+'-bound.pdf', format='pdf')
  # Show the plot
  # plt.xlim(-1, 3)
  plt.show()

  plt.figure(figsize=(6, 4))
  plt.plot(sample,mi_avg, marker = '+',markersize = 8, color='orange', label="MI")
  plt.plot(sample,hd_avg, color='red', label="HD")
  plt.plot(sample,jsd_avg, color='blue', label="JSD")
  plt.plot(sample,chi_avg, color='yellow', label="Chi^2")
  # plt.plot(sample,fcmi_bound, label="fCMI")
  # plt.plot(sample,sub_bound, color='yellow', label="Sub")
  # plt.plot(sample,err, color='gray', label="Err")
  plt.legend()
  plt.savefig(exp_name + '-divergence.pdf', format='pdf')
  plt.show()

  # plt.figure()
  # plt.plot(sample,opt_bound, color='pink', label="Opt")
  # plt.plot(sample,var_bound, color='green', label="Var")
  # plt.plot(sample,sharp_bound, marker = 'x',markersize = 8, label="Sharp")
  # plt.plot(sample,kl_bound, marker = '2',markersize = 8, label="KL")
  # plt.plot(sample,err, color='gray', label="Err")
  # plt.legend()

  # plt.figure()
  # plt.plot(sample,single_ip_bound, color='cyan', label="SIP-bound")
  # plt.plot(sample,pair_ip_bound, color='magenta', label="PIP-bound")
  # plt.plot(sample,exact_bound, label="Perfect-bound")
  # plt.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.set_defaults(parse=True)
    args = parser.parse_args()
    print(args)

    if args.exp_name in ["two-simple"]:
        sample = [10, 20, 30, 50, 80, 100]
        PlotBound(n_class=2, dim=5, diff=5, num_Z=100, num_U=200, sample=sample, exp_name=args.exp_name)
    elif args.exp_name in ["two-normal"]:
        sample = [10, 20, 30, 50, 80, 100, 200]
        PlotBound(n_class=2, dim=5, diff=1, num_Z=100, num_U=200, sample=sample, exp_name=args.exp_name)
    elif args.exp_name in ["two-hard"]:
        sample = [10, 20, 30, 50, 80, 100, 200, 300]
        PlotBound(n_class=2, dim=5, diff=0.2, num_Z=100, num_U=200, sample=sample, exp_name=args.exp_name)
    elif args.exp_name in ["ten-simple"]:
        sample = [10, 20, 30, 50, 80, 100]
        PlotBound(n_class=10, dim=5, diff=5, num_Z=50, num_U=100, sample=sample, exp_name=args.exp_name)
    elif args.exp_name in ["ten-normal"]:
        sample = [10, 20, 30, 50, 80, 100, 200]
        PlotBound(n_class=10, dim=5, diff=1, num_Z=50, num_U=200, sample=sample, exp_name=args.exp_name)
    elif args.exp_name in ["ten-hard"]:
        sample = [10, 20, 30, 50, 80, 100, 200]
        PlotBound(n_class=10, dim=5, diff=0.2, num_Z=50, num_U=200, sample=sample, exp_name=args.exp_name)
    else:
        raise ValueError(f"Unexpected exp_name: {args.exp_name}")

    # exp_name = "fcmi-mnist-4vs9-CNN"



if __name__ == '__main__':
    main()