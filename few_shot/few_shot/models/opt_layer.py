import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import gradcheck
import numpy as np
from scipy import optimize
from scipy.optimize import check_grad
from scipy.sparse.linalg import LinearOperator, cg, cgs, bicg, bicgstab, gmres, lgmres, minres, lsqr, lsmr
from numpy.linalg import norm, svd, inv, eig
from models.SPG import *


def optsolver(K_batch, Phi, a, imp, exc ,gamma):
    batch_size, nfea = K_batch.size()
    #nsample,nfea = Phi.size()
    K_batch = K_batch.numpy()
    #AB = AB.numpy()
    Phi = Phi.numpy()
    imp = imp.numpy()
    exc = exc.numpy()
    imp = norm(imp)**2
    exc = norm(exc)**2
    a = a.numpy()
    #a = AB[0][0]
    #norm_AB = norm(AB)
    def obj(x, gamma=gamma):
        Rx = x.reshape((batch_size, nfea))
        #print('Rx', norm(Rx[0])**2)
        f =(gamma * (norm(Rx)**2)) / batch_size      # batch_size * norm_AB**2
        #print('f1', f)
        #print('Rx', Rx)
        #print('f', f)
        #RPhi = Rx @ Phi.T 
        A = (imp + exc)/batch_size * Phi
        #print('Phi_size', Phi.shape)
        #print('rx_size', Rx.shape)
        #print('A_size', A.shape)
        f = f + (np.sum(Rx @ A *Rx))
        #print('f2', f)
        g = gamma/ batch_size * 2 * Rx
        g = g + 2* Rx @ A
        g = g.flatten()
        return f, g

    def proj(x, K_batch, a):
        x = x.reshape((batch_size, nfea))
        lam = (1.0/a - np.sum(x * K_batch, axis = 1))/ np.sum(K_batch * K_batch, axis = 1)
        return (x + np.multiply(K_batch, lam[:, np.newaxis])).flatten()

   
    def con_fun(x):
        Rx = x.reshape((batch_size, nfea))
        RK = np.sum((Rx * K_batch),axis = 1)
        return a * RK - np.ones_like(RK)

    def con_jac(x):
        print('incon')
        jac = np.zeros((batch_size, batch_size*nfea))
        for i in range(batch_size):
            jac[i][nfea*i : nfea*(i+1)] = a * K_batch[i]
        print('outcon')
        return jac
    '''
    x0 = (K_batch +1).flatten()
    npa = np.array([2.4, 2.9])
    print('np type', npa.dtype)
    #print('x0', x0-npa)
    err = check_grad(obj, grad, x0, gamma, epsilon = 1e-8)
    print('err',err)
    eps = np.sqrt(np.finfo(float).eps)
    gg = optimize.approx_fprime(x0, obj, 1e-8, gamma)
    print('gg',gg)
    def func(x):
        return norm(x)**2#x[0]**2 +  x[1]**2
    def grad1(x):
        g = 2 * x 
        #print('res:', g)
        return g#[2 * x[0], 2 * x[1]]
    print('####', check_grad(func, grad1, npa))
    '''
    
    x0 = K_batch.flatten()
    
    spg_options = default_options
    default_options.verbose = 0
    funObj = lambda x: obj(x, gamma)
    funProj = lambda x: proj(x, K_batch, a)
    opt_x, opt_f = SPG(funObj, funProj, x0, spg_options)
    opt_x = opt_x.reshape(batch_size, nfea)
    
    '''
    eq_cons = {'type' : 'eq', 
               'fun' : con_fun,
               'jac' : con_jac}
    res = optimize.minimize(obj, x0, method = 'SLSQP', jac = True, constraints = [eq_cons], 
                            options = {'ftol': 1e-8, 'disp': True})

    opt_x = res['x']
    opt_x = opt_x.reshape(batch_size, nfea)
    #opt_obj = res['fun']
    '''
    dr1 = (gamma * 2 / batch_size) * opt_x + (2/batch_size) * (imp + exc) * opt_x @ Phi
    dr2 = a * K_batch
    
    opt_lambda = -1 * np.sum(dr1 * dr2, axis = 1) / np.sum(dr2 * dr2, axis = 1)
    #print('opt_x', opt_x)
    #print('k_batch', K_batch)
    
    return torch.from_numpy(opt_x.astype(np.float32)), torch.from_numpy(opt_lambda.astype(np.float32))

def inv_A(dLdR, K_batch, a, Phi, gamma, imp, exc):
    #global i 
    i = torch.ones((1,1))
    batch_size, nfea = K_batch.size()
    #nsample,nfea = Phi.size()
    K_batch = K_batch.numpy()
    dLdR = dLdR.numpy()
    #AB = AB.numpy()
    Phi = Phi.numpy()
    imp = imp.numpy()
    exc = exc.numpy()
    imp = norm(imp)**2
    exc = norm(exc)**2
    a = a.numpy()
    #a = AB[0][0]
    #norm_AB = norm(AB)
    e = 1e-8
    K_Phi = 2 * (imp + exc)/batch_size * Phi + (2*gamma/batch_size) * np.identity(nfea)
    aK_batch = a * K_batch
    aK_batch_s = np.sum(aK_batch * aK_batch, axis = 1)
    K_bK_Phi = K_batch @ K_Phi
    K_Phi_s = K_Phi @ K_Phi.T
    dLdR_aK = np.sum(dLdR * aK_batch, axis = 1)
    dLdR_KP = dLdR @ K_Phi
    b = np.concatenate((dLdR_KP.flatten(), dLdR_aK.flatten()), axis=0)/batch_size
    def obj_inv_A(x, dLdR=dLdR):
        dR = x[:batch_size*nfea].reshape((batch_size, nfea))
        dlam = x[batch_size*nfea:]
        dR_aK = np.sum(dR * aK_batch, axis = 1)
        dR_KbKP = np.sum(dR * K_bK_Phi, axis = 1)
        f = np.sum(dR @ K_Phi_s * dR) + norm(dR_aK)**2 ##
        #f = f + gamma/batch_size * norm(dR)**2
        f = 0.5 * f + a * np.sum((np.squeeze(dR_KbKP) * dlam)) ##
        f = f + 0.5 * np.sum(aK_batch_s * (dlam * dlam))##
        f = f -  (np.sum(dLdR_KP * dR) + np.sum(dLdR_aK * dlam)) ##
        gr = dR @ K_Phi_s + np.multiply(aK_batch, dR_aK[:, np.newaxis])
        gr = gr + a * np.multiply(K_bK_Phi, dlam[:, np.newaxis])
        gr = gr - dLdR_KP 
        #gr = np.zeros((batch_size, nfea))
        gl = a * dR_KbKP
        gl = gl + dlam * aK_batch_s
        gl = gl - dLdR_aK
        #gl = np.zeros(batch_size)
        g = np.concatenate((gr.flatten(), gl.flatten()), axis=0)
        #print('g', gr)
        return f, g
    '''
    def obj_test(x, dLdR=dLdR):
        H = np.block([[2*K_Phi, a*K_batch.T],[a*K_batch, e]])
        f = 0.5 * x.T @ H @ H.T @ x
        f = f - np.sum((H @ x)[:batch_size*nfea] * dLdR.flatten())
        L = np.zeros(nfea+1)
        L[:nfea] = dLdR.flatten()
        g = H @ H.T @ x - H @ L
        return f
    '''
    def Hx(x):
        #print('i')
        #print('x', x[0:5])
        dR = x[:batch_size*nfea].reshape((batch_size, nfea))/batch_size
        dlam = x[batch_size*nfea:]/batch_size
        dR_aK = np.sum(dR * aK_batch, axis = 1)
        dR_KbKP = np.sum(dR * K_bK_Phi, axis = 1)
        #print('dR0', dR[0][0:5])
        gr = dR @ K_Phi_s + np.multiply(aK_batch, dR_aK[:, np.newaxis])
        
        gr = gr + a * np.multiply(K_bK_Phi, dlam[:, np.newaxis])
        gr = gr + e * dR
        #gr = gr - dLdR_KP 
        gl = a * dR_KbKP
        gl = gl + dlam * aK_batch_s
        gl = gl + e * dlam
        #gl = gl - dLdR_aK
        g = np.concatenate((gr.flatten(), gl.flatten()), axis=0)
        #print('g',g[0:5])
        return g*1000

    def grad_inv_A(x, dLdR=dLdR):
        dR = x[:batch_size*nfea].reshape((batch_size, nfea))
        dlam = x[batch_size*nfea:]
        f = np.sum(dR @ (K_Phi @ K_Phi.T ) * dR) + norm(np.sum(dR * (a * K_batch), axis = 1))**2 ##
        #f = f + gamma/batch_size * norm(dR)**2
        f = 0.5 * f + a * np.sum((np.squeeze(np.sum(dR* (K_batch @ K_Phi), axis = 1)) * dlam)) ##
        f = f + 0.5 * np.sum(np.sum((a*K_batch) * (a*K_batch), axis = 1) * (dlam * dlam))##
        print('dldr', dLdR)
        print('dr', dR.shape)
        print('K_phi', K_Phi.shape)
        f = f -  (np.sum(dLdR @ K_Phi * dR) + np.sum(np.sum(dLdR * a * K_batch, axis = 1) * dlam)) ##
        gr = dR @ (K_Phi @ K_Phi.T) + np.multiply(a * K_batch, np.sum(dR * (a * K_batch), axis = 1)[:, np.newaxis])
        gr = gr + a * np.multiply(K_batch @ K_Phi, dlam[:, np.newaxis])
        gr = gr - dLdR @ K_Phi 
        #gr = np.zeros((batch_size, nfea))
        gl = a * np.sum(dR * (K_batch @ K_Phi), axis = 1)
        gl = gl + dlam * np.sum((a*K_batch) * (a*K_batch), axis = 1)
        gl = gl -  np.sum(dLdR * a * K_batch, axis = 1)
        #gl = np.zeros(batch_size)
        g = np.concatenate((gr.flatten(), gl.flatten()), axis=0)
        #print('g', gr)
        return g
    '''
    def grad1_inv_A(x, dLdR=dLdR):
        dR = x[:batch_size*nfea].reshape((batch_size, nfea))
        dlam = x[batch_size*nfea:]
        f = np.sum(dR @ K_Phi * dR)
        f = f + gamma/batch_size * norm(dR)**2
        f = f + a * np.sum((np.squeeze(np.sum(dR*K_batch, axis = 1)) * dlam))
        f = f + 0.5 * norm(dlam)**2 * e
        f = f -  np.sum(dLdR * dR)
        gr = 2 * dR @ K_Phi
        gr = gr + 2 * gamma/batch_size * dR
        gr = gr + a * np.multiply(K_batch, dlam[:, np.newaxis])
        gr = gr - dLdR
        gl = a * np.sum(dR * K_batch, axis = 1)
        gl = gl + dlam * e
        #gl = np.zeros(batch_size)
        g = np.concatenate((gr.flatten(), gl.flatten()), axis=0)
        return g
    '''
    l0 = np.zeros(batch_size)
    r0 = dLdR.flatten()
    x0 = np.concatenate((r0, l0), axis=0)
    #err = check_grad(obj_inv_A, grad_inv_A, x0, dLdR, epsilon = 1e-8)
    #print('err',err)
    #print('before')
    A = LinearOperator((batch_size*nfea+batch_size, batch_size*nfea+batch_size), matvec = Hx)
    res = minres(A, b*1000, tol = 1e-8, maxiter =200, x0 = x0)
    opt_x = res[0]
    #print('res', res[1])
    #print('i', i)
    #print('opt_x1', opt_x)
    #res = optimize.minimize(obj_inv_A, x0, method = 'L-BFGS-B', jac = True, options = {'ftol':1e-16,'gtol': 1e-16, 'disp': False})
    #opt_x = res['x']
    #print('opt_x2', opt_x)
    #print('after')
    
    dr = -1.0 * opt_x[:batch_size*nfea].reshape((batch_size, nfea))
    dlam = -1.0 * opt_x[batch_size*nfea:]
    #print('opt_x', opt_x[:batch_size*nfea].reshape((batch_size, nfea)))
    #print('opt_x', opt_x[:batch_size*nfea].reshape((batch_size, nfea)))
    '''
    inv_x = np.zeros((batch_size, nfea))
    for i in range(1):
        dldr = dLdR[i]
        ki = K_batch[i]
        #print('ki', ki[:, np.newaxis].shape)
        L = np.zeros(nfea+1)
        L[:nfea] = dldr
        H = np.block([[K_Phi,a*ki[:, np.newaxis]],[a*ki[:, np.newaxis].T, e]])
        #print('h', H.shape)
        inv_x[i] = (inv(H) @ L)[:nfea]
    print('inv_x', inv_x)
    print('dr_in', dr)
    '''
    '''
    fval = 0.0
    for i in range(1):
        rx = dr[i]
        rlam = dlam[i]
        r = np.zeros(nfea+1)
        r[:nfea] = rx
        r[-1] = rlam
        r = -1.0 * r
        print('rx', rx)
        print('rlam', rlam)
        print('r', r)
        dldr = dLdR[i]
        ki = K_batch[i]
        #print('ki', ki[:, np.newaxis].shape)
        L = np.zeros(nfea+1)
        L[:nfea] = dldr
        H = np.block([[K_Phi,a*ki[:, np.newaxis]],[a*ki[:, np.newaxis].T, e]])
        #print('h', H.shape)
        #inv_x[i] = (inv(H) @ L)[:nfea]
        fval += 0.5 * r.T @ H.T @ H @ r - r.T @ H @ L
    print('inv_f', fval)
    print('f', res['fun'])
    '''
    
    ki = K_batch[0]
    H = np.block([[K_Phi,a*ki[:, np.newaxis]],[a*ki[:, np.newaxis].T, 0]])
    lr = dLdR[0]
    L = np.zeros(nfea+1)
    L[:nfea] = lr
    print('inv', -1 * (inv(H + e* np.eye(nfea+1)) @ L)[0:5])
    print('dr', dr[0][0:5])
    print('dldr', lr[0:5])
    
    #opt_obj = res['fun']
    return torch.from_numpy(dr.astype(np.float32)), torch.from_numpy(dlam.astype(np.float32))

    
   
class opt_layer(Function):

    @staticmethod
    def forward(ctx, K_batch, a, exc, imp, Phi, gamma, device):
        K = K_batch.detach().cpu()
        P = Phi.detach().cpu()
        a = a.detach().cpu()
        i = imp.detach().cpu()
        e = exc.detach().cpu()
        opt_x, opt_lambda = optsolver(K, P, a, i, e, gamma)
        ctx.device = device
        ctx.gamma = gamma
        ctx.save_for_backward(opt_x, opt_lambda, K, e, i, a, P)

        return torch.as_tensor(opt_x).to(device)

    @staticmethod
    def backward(ctx, dLdR):
        
        #print('dLdR', dLdR)
        device = ctx.device
        gamma = ctx.gamma
        opt_x, opt_lam, K_batch, exc, imp, a, Phi = ctx.saved_tensors
        [batch_size, nfea] = K_batch.size()
        #[nsample, nfea] = Phi.size()
        dldr = dLdR.detach().cpu().clone()
        dr, dlam = inv_A(dldr, K_batch, a, Phi, gamma, imp, exc)
        '''
        print('dr:', dr)
        p = np.zeros(nfea+1)
        p[:nfea] = dLdR.detach().numpy()
        dd = inv(H) @ p
        dr1 = torch.from_numpy(dd[:nfea])
        dlam1 = torch.from_numpy(dd[nfea:])
        print('dr1', dr1)
        '''
        #dr, dlam, H = inv_A(dLdR, K_batch, a, Phi, gamma, imp, exc)
        dK_batch = a * dr * opt_lam[:, None] + a * opt_x * dlam[:, None]
        #print('dr', dr)
        dexc = (4/batch_size) * torch.sum((opt_x @ Phi) * dr) * exc
        dimp = (4/batch_size) * torch.sum((opt_x @ Phi) * dr) * imp   ##
        da = torch.sum(K_batch * dr, dim=1) @ opt_lam  + torch.sum(K_batch * opt_x, dim=1) @ dlam
        return torch.as_tensor(dK_batch).to(device), torch.as_tensor(da).to(device), torch.as_tensor(dexc).to(device), torch.as_tensor(dimp).to(device), None, None, None


class OptNet(nn.Module):
    def __init__(self, config):
        super(OptNet, self).__init__()
        self.device = config['device']
        self.gamma = config['gamma']
        self.a = nn.Parameter(data = torch.tensor(config['a_value']), requires_grad=True)
        self.nclass = config['nclass']
        self.beta = nn.Parameter(data = torch.zeros((self.nclass, self.nclass)), requires_grad=True)
        self.exc_relation = config['exc'].to(self.device)
        self.imp_relation = config['imp'].to(self.device)
        self.eye = torch.eye(self.nclass, self.nclass).to(self.device)
        self.Phi = config['Phi']
        self.opt_layer = opt_layer.apply
        

        

    def forward(self, x):
        #mask = torch.eye(self.nclass, self.nclass).byte()
        #AB = self.beta.masked_fill_(mask, 0) + self.a * torch.eye(self.nclass, self.nclass)
        #print('x_size', x.size())
        #print('x_norm', torch.norm(x[0]))
        #print('Phi_in', Phi.is_cuda)
        #print('norm_in', torch.norm(x[0]))
        AB = self.beta + self.a * self.eye
        exc = torch.max(AB @ self.exc_relation, dim=1)[0]
        imp = torch.max(AB @ self.imp_relation, dim=1)[0]
        rx = self.opt_layer(x, self.a, exc, imp, self.Phi, self.gamma, self.device)
        #print('norm_out', torch.norm(rx[0]))
        return rx






        

'''
#K_batch, AB, Phi, gamma, imp, exc
batch_size = 3
nfea = 1000
nclass =4
nsample = 6
K_batch = torch.rand((batch_size, nfea), requires_grad=True)/10 #dtype = torch.float64
print('normk', torch.norm(K_batch[0])**2)
dLdR = torch.rand((batch_size, nfea))/1e8
#dLdR = torch.zeros((batch_size, nfea))
#sdLdR[0][0] = 1.0
#K_batch = torch.tensor([[1.4,1.9]], dtype = torch.float64)
print('torch type',K_batch.dtype)
a = torch.tensor(0.1, requires_grad=True)
#AB = torch.rand((nclass, nclass), dtype = torch.float64).fill_diagonal_(a)
Phi = torch.rand((nsample,nfea))/10
print('Phi', Phi.size())
gamma = 1.0
imp = torch.rand((nclass, 1),requires_grad=True)
exc = torch.rand((nclass, 1),requires_grad=True)
#opt_x, opt_lam = optsolver(K_batch.detach(), Phi.detach(), a.detach(), imp.detach(), exc.detach(), gamma)
#dr, dlam = inv_A(dLdR, K_batch, a, Phi, gamma, imp, exc)
#print('init', K_batch.size())
#opt_x, opt_lam = optsolver(K_batch.detach(), (torch.transpose(Phi, 0, 1) @ Phi / nsample).detach(), a.detach(), imp.detach(), exc.detach(), gamma)
#a1 = (a.detach()*opt_lam[0]*K_batch.detach()[0][0]).numpy()
#a2 = (a.detach()*K_batch.detach()[0][0] * opt_x[0][0]).numpy()
#tensor = [a1, 0.0, 0.0, 0.0, 0.0, a2]
#tensor = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dr, dlam, H = inv_A(dLdR, K_batch.detach(), a.detach(), (torch.transpose(Phi, 0, 1) @ Phi / nsample).detach(), gamma, imp.detach(), exc.detach())
print('dr', dr[0])
lr = dLdR[0].numpy()
L = np.zeros(nfea+1)
L[:nfea] = lr
print('inv', inv(H) @ L)
#L = np.block([[a.detach().numpy() * opt_lam.numpy() * np.identity(nfea)],[a.detach().numpy() * opt_x.numpy()]])
#print(L)
#L = np.array(tensor)
#print('inv', inv(H) @ L)
#L = np.concatenate((opt_x.numpy().flatten(),opt_lam.numpy().flatten()), axis=0)
#print('opt', H@L)
#input = (K_batch, a, exc, imp, (torch.transpose(Phi, 0, 1) @ Phi / nsample), gamma, 'cpu')
#err = gradcheck(opt_layer.apply, input, eps=1e-5, atol=1e-3)
'''
'''
for i in range(3):
        dldr = dLdR[i]
        ki = K_batch[i]
        #print('ki', ki[:, np.newaxis].shape)
        L = np.zeros(nfea+1)
        L[:nfea] = dldr
        H = np.block([[K_Phi,a*ki[:, np.newaxis]],[a*ki[:, np.newaxis].T, e]])
        #print('h', H.shape)
        inv_x[i] = (inv(H) @ L)[:nfea]
    print('inv_x', inv_x)
'''



