"""TP débutant : modèles et optimiseurs écrits avec NumPy.
Exécution : python tp.py --epochs 150
Les fichiers CSV supplémentaires et les NPZ du sujet sont détectés automatiquement.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, csv, json, time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'resultats' / 'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, confusion_matrix, adjusted_rand_score,
    normalized_mutual_info_score, silhouette_score)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultats'
OUT.mkdir(exist_ok=True)
METHODS = ['GD', 'GD-recherche', 'SGD', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam']
SEEDS = [42, 123, 2024]

def distances(x, c):
    return np.maximum((x*x).sum(1)[:,None] + (c*c).sum(1)[None,:] - 2*x@c.T, 0)

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

class Regression:
    def __init__(self, x, y, model, lam=0., sigma=1.):
        self.x, self.y, self.model, self.lam, self.sigma = x, y, model, lam, sigma
        self.a = self.features(x)
        self.r = self.a.copy() if model == 'RBF' else np.diag([0.] + [1.]*x.shape[1])
        self.initial = np.zeros(self.a.shape[1])
        self.h = self.a.T@self.a/len(x) + lam*self.r
    def features(self, x):
        if self.model == 'RBF': return np.exp(-distances(x,self.x)/(2*self.sigma**2))
        return np.column_stack([np.ones(len(x)), x])
    def loss(self, p, x=None, y=None):
        a, y = (self.a,self.y) if x is None else (self.features(x),y)
        return float(np.mean((a@p-y)**2)/2 + self.lam*(p@self.r@p)/2)
    def grad(self, p, indices):
        a = self.a[indices]
        return a.T@(a@p-self.y[indices])/len(indices) + self.lam*self.r@p
    def predict(self, p, x): return self.features(x)@p

class Network:
    def __init__(self, x, y, seed, classes, images=False):
        self.x,self.y,self.lam = x,y,1e-4
        dims = [x.shape[1],128,64,classes] if images else [x.shape[1],32,16,classes]
        rng = np.random.default_rng(seed)
        self.shapes = []
        parts = []
        for a,b in zip(dims[:-1],dims[1:]):
            self.shapes.extend([(a,b),(b,)])
            parts.extend([rng.normal(0,np.sqrt(2/(a+b)),(a,b)).ravel(),np.zeros(b)])
        self.initial = np.concatenate(parts)
    def unpack(self,p):
        result, pos = [],0
        for shape in self.shapes:
            size = int(np.prod(shape)); result.append(p[pos:pos+size].reshape(shape)); pos += size
        return result
    def forward(self,p,x):
        w1,b1,w2,b2,w3,b3 = self.unpack(p)
        h1=np.tanh(x@w1+b1); h2=np.tanh(h1@w2+b2)
        return h1,h2,softmax(h2@w3+b3)
    def loss(self,p,x=None,y=None):
        if x is None: x,y=self.x,self.y
        prob=self.forward(p,x)[2]
        return float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-300)).mean()+self.lam/2*sum((w*w).sum() for w in self.unpack(p)[::2]))
    def grad(self,p,indices):
        x,y=self.x[indices],self.y[indices]
        w1,b1,w2,b2,w3,b3=self.unpack(p)
        h1,h2,prob=self.forward(p,x)
        d3=prob.copy(); d3[np.arange(len(y)),y]-=1; d3/=len(y)
        d2=(d3@w3.T)*(1-h2*h2); d1=(d2@w2.T)*(1-h1*h1)
        return np.concatenate([a.ravel() for a in [x.T@d1+self.lam*w1,d1.sum(0),h1.T@d2+self.lam*w2,d2.sum(0),h2.T@d3+self.lam*w3,d3.sum(0)]])
    def predict(self,p,x): return self.forward(p,x)[2]

class Clustering:
    def __init__(self,x,seed,k=3,tau=.25):
        self.x,self.k,self.tau=x,k,tau
        self.initial=x[np.random.default_rng(seed).choice(len(x),k,replace=False)].ravel().copy()
    def loss(self,p,x=None,y=None):
        if x is None: x=self.x
        z=-distances(x,p.reshape(self.k,-1))/self.tau
        m=z.max(1)
        return float(-self.tau*np.mean(m+np.log(np.exp(z-m[:,None]).sum(1))-np.log(self.k)))
    def grad(self,p,indices):
        x=self.x[indices]; c=p.reshape(self.k,-1)
        q=softmax(-distances(x,c)/self.tau)
        return (2/len(x)*(q.sum(0)[:,None]*c-q.T@x)).ravel()
    def predict(self,p,x): return distances(x,p.reshape(self.k,-1)).argmin(1)

def check_gradient(model):
    p=model.initial.copy()+np.random.default_rng(7).normal(0,.03,len(model.initial))
    indices=np.arange(min(11,len(model.x)))
    # Pour le noyau, le terme de régularisation reste calculé sur tout K.
    def batch_loss(v):
        if isinstance(model,Regression):
            e=model.a[indices]@v-model.y[indices]
            return np.mean(e*e)/2+model.lam/2*(v@model.r@v)
        return model.loss(v,model.x[indices],None if isinstance(model,Clustering) else model.y[indices])
    g=model.grad(p,indices); errors=[]
    for j in np.linspace(0,len(p)-1,min(12,len(p)),dtype=int):
        a=p.copy(); b=p.copy(); a[j]+=1e-5; b[j]-=1e-5
        numeric=(batch_loss(a)-batch_loss(b))/2e-5
        errors.append(abs(numeric-g[j])/max(1e-8,abs(numeric)+abs(g[j])))
    return float(max(errors))

def train(model,method,eta,epochs,seed,xval,yval,batch):
    p=model.initial.copy(); m=np.zeros_like(p); s=m.copy(); v=m.copy()
    rng=np.random.default_rng(seed); history=[]; updates=0; losses=0; best=np.inf
    bestp=p.copy(); bestepoch=0; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        order=np.arange(len(model.x)) if method.startswith('GD') else rng.permutation(len(model.x))
        size=len(order) if method.startswith('GD') else batch
        for begin in range(0,len(order),size):
            g=model.grad(p,order[begin:begin+size]); updates+=1
            if method=='GD-recherche':
                if isinstance(model,Regression):
                    denominator=g@model.h@g
                    step=(g@g)/denominator if denominator>1e-20 else 0.
                else:
                    # Recherche numérique bornée par section dorée, 10 itérations.
                    left,right=0.,eta; ratio=(np.sqrt(5)-1)/2
                    a=right-ratio*(right-left); b=left+ratio*(right-left)
                    fa=model.loss(p-a*g); fb=model.loss(p-b*g); losses+=2
                    for _ in range(10):
                        if fa<fb:
                            right,b,fb=b,a,fa; a=right-ratio*(right-left); fa=model.loss(p-a*g)
                        else:
                            left,a,fa=a,b,fb; b=left+ratio*(right-left); fb=model.loss(p-b*g)
                        losses+=1
                    step=(left+right)/2
                    current=model.loss(p); trial=model.loss(p-step*g); losses+=2
                    if trial>current: step=0.
                p-=step*g
            elif method in ['GD','SGD']: p-=eta*g
            elif method=='Momentum': v=.9*v+g; p-=eta*v
            elif method=='AdaGrad': s+=g*g; p-=eta*g/(np.sqrt(s)+1e-8)
            elif method=='RMSprop': s=.9*s+.1*g*g; p-=eta*g/(np.sqrt(s)+1e-8)
            else:
                m=.9*m+.1*g; s=.999*s+.001*g*g
                p-=eta*(m/(1-.9**updates))/(np.sqrt(s/(1-.999**updates))+1e-8)
        loss=model.loss(p); val=model.loss(p,xval,yval); losses+=2
        score=val
        extra={}
        if isinstance(model,Regression):
            score=np.sqrt(np.mean((model.predict(p,xval)-yval)**2)); extra['rmse_val_standardisee']=score
        elif isinstance(model,Network):
            extra={'accuracy_train':float(np.mean(model.predict(p,model.x).argmax(1)==model.y)),
                   'accuracy_val':float(np.mean(model.predict(p,xval).argmax(1)==yval))}
        history.append(dict(epoque=epoch,temps=time.perf_counter()-start,perte_train=loss,perte_val=val,**extra))
        if np.isfinite(score) and score<best:
            best,bestp,bestepoch=score,p.copy(),epoch
    return bestp,history,dict(epoques=epochs,meilleure_epoque=bestepoch,temps=time.perf_counter()-start,mises_a_jour=updates,appels_gradient=updates,appels_perte=losses,selection=best)

def read_csv(path):
    data=np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf-8')
    columns=[n for n in data.dtype.names if n not in ['y','label','true_label','split']]
    x=np.column_stack([data[n] for n in columns]).astype(float)
    target=next((n for n in ['y','label','true_label'] if n in data.dtype.names),None)
    y=data[target] if target else None
    if 'split' not in data.dtype.names: return x,y,None
    return x,y,data['split']

def save_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def evaluate(model,p,x,y,ym=0,ys=1):
    pred=model.predict(p,x)
    if isinstance(model,Regression):
        truth=y*ys+ym; pred=pred*ys+ym; mse=mean_squared_error(truth,pred)
        return dict(MSE=mse,RMSE=np.sqrt(mse),MAE=mean_absolute_error(truth,pred),R2=r2_score(truth,pred))
    if isinstance(model,Network):
        labels=pred.argmax(1)
        pr,re,f1,_=precision_recall_fscore_support(y,labels,average='macro',zero_division=0)
        return dict(exactitude=np.mean(labels==y),precision_macro=pr,rappel_macro=re,F1_macro=f1,entropie_test=-np.log(np.maximum(pred[np.arange(len(y)),y],1e-300)).mean())
    d=distances(x,p.reshape(model.k,-1)); nonempty=len(np.unique(pred))
    idx=np.random.default_rng(99).choice(len(x),min(1000,len(x)),replace=False)
    labels=pred[idx]
    sil=silhouette_score(x[idx],labels) if 1<len(np.unique(labels))<len(idx) else np.nan
    return dict(J=model.loss(p,x),inertie=d.min(1).mean(),clusters_non_vides=nonempty,silhouette=sil,ARI=adjusted_rand_score(y,pred),NMI=normalized_mutual_info_score(y,pred))

def run_dataset(path,epochs,results,checks):
    name=path.stem; images=path.suffix=='.npz'
    kinds=['classification','clustering'] if images else [name.split('_')[0]]
    for kind in kinds:
        if images:
            z=np.load(path); xt=z['x_train'].reshape(-1,784).astype(np.float32)/255
            xv=z['x_val'].reshape(-1,784).astype(np.float32)/255; xe=z['x_test'].reshape(-1,784).astype(np.float32)/255
            yt,yv,ye=z['y_train'].astype(int),z['y_val'].astype(int),z['y_test'].astype(int)
            if kind=='classification':
                mean=xt.mean(0); xt,xv,xe=xt-mean,xv-mean,xe-mean
        else:
            x,y,split=read_csv(path)
            if kind=='clustering': xt=x.copy(); xv=x.copy(); xe=x.copy(); yt=yv=ye=y
            else:
                xt,xv,xe=[x[split==s] for s in ['train','val','test']]
                yt,yv,ye=[y[split==s] for s in ['train','val','test']]
            mean,std=xt.mean(0),xt.std(0); std=np.where(std==0,1,std)
            xt,xv,xe=[(a-mean)/std for a in [xt,xv,xe]]
        ym,ys=0,1
        if kind=='regression':
            ym,ys=float(yt.mean()),float(yt.std()); yt,yv,ye=[(a-ym)/ys for a in [yt,yv,ye]]
        models=['OLS','Ridge','RBF'] if kind=='regression' else [kind]
        for modelname in models:
            lam,sigma=0.,1.
            if modelname in ['Ridge','RBF']:
                candidates=[]
                for l in [.001,.01,.1]:
                    for sig in ([.5,1.,2.] if modelname=='RBF' else [1.]):
                        control=Regression(xt,yt,modelname,l,sig)
                        solution=np.linalg.solve(control.a+len(xt)*l*np.eye(len(xt)),yt) if modelname=='RBF' else np.linalg.solve(control.h,control.a.T@yt/len(xt))
                        candidates.append((np.mean((control.predict(solution,xv)-yv)**2),l,sig))
                _,lam,sigma=min(candidates)
            def make(seed):
                if kind=='regression': return Regression(xt,yt,modelname,lam,sigma)
                if kind=='classification': return Network(xt,yt.astype(int),seed,10 if images else 3,images)
                return Clustering(xt,seed,10 if images else 3,5 if images else .25)
            base=make(42); error=check_gradient(base)
            checks.append(dict(jeu=name,modele=modelname,erreur_relative=error))
            if error>1e-3: raise ValueError('Gradient incorrect : '+name+' '+modelname)
            comparison=[]; selected=[]
            for method in METHODS:
                grid=[.01,.1] if method in ['GD','SGD'] else ([1.,5.] if method=='GD-recherche' else [.001,.01])
                if kind=='clustering': grid=[.01,.1] if method!='GD-recherche' else [1.,5.]
                if kind=='regression' and modelname=='RBF' and method in ['GD','SGD']: grid=[.001,.01]
                batch=1024 if images and kind=='classification' else (2048 if images else 32)
                trials=[]
                for eta in grid:
                    pp,hh,cc=train(make(42),method,eta,epochs,42,xv,yv,batch)
                    trials.append((cc['selection'],eta,pp,hh,cc))
                _,eta,p,h,c=min(trials,key=lambda a:a[0])
                for seed in SEEDS:
                    model=make(seed)
                    if seed!=42: p,h,c=train(model,method,eta,epochs,seed,xv,yv,batch)
                    metrics=evaluate(model,p,xe,ye,ym,ys)
                    row=dict(jeu=name,exercice=kind,modele=modelname,methode=method,graine=seed,eta=eta,lambda_=lam,sigma=sigma if modelname=='RBF' else '',tau=model.tau if kind=='clustering' else '',**c,**metrics)
                    if kind=='regression':
                        row['RMSE_validation']=c['selection']*ys
                        reference=np.linalg.lstsq(model.a,yt,rcond=None)[0] if lam==0 else (np.linalg.solve(model.a+len(xt)*lam*np.eye(len(xt)),yt) if modelname=='RBF' else np.linalg.solve(model.h,model.a.T@yt/len(xt)))
                        row['ecart_predictions_controle']=float(np.sqrt(np.mean((model.predict(p,xt)-model.predict(reference,xt))**2)))
                    results.append(row)
                    save_csv(OUT/f'{name}_{modelname}_{method}_{seed}_courbes.csv',h)
                    if seed==42:
                        comparison.append((method,h)); selected.append((c['selection'],model,p,method))
            prefix=f'{name}_{modelname}'
            fig,axes=plt.subplots(1,3,figsize=(12,3))
            for method,h in comparison:
                axes[0].plot([a['epoque'] for a in h],[a['perte_train'] for a in h],label=method)
                axes[1].plot([a['epoque'] for a in h],[a['perte_val'] for a in h],label=method)
                axes[2].plot([a['temps'] for a in h],[a['perte_val'] for a in h],label=method)
            for ax,title in zip(axes,['Entraînement','Validation','Validation selon le temps']): ax.set_title(title); ax.set_ylabel('Objectif'); ax.set_xlabel('Secondes' if ax==axes[2] else 'Époque')
            axes[0].legend(fontsize=6); fig.suptitle(prefix+' - graine 42'); fig.tight_layout(); fig.savefig(OUT/(prefix+'_courbes.png'),dpi=140); plt.close(fig)
            _,model,p,method=min(selected,key=lambda a:a[0])
            if kind in ['regression','classification']:
                fig,axes=plt.subplots(1,2,figsize=(9,3))
                for method_curve,hcurve in comparison:
                    key='rmse_val_standardisee' if kind=='regression' else 'accuracy_val'
                    factor=ys if kind=='regression' else 1
                    axes[0].plot([a['epoque'] for a in hcurve],[a[key]*factor for a in hcurve],label=method_curve)
                    axes[1].plot([a['temps'] for a in hcurve],[a[key]*factor for a in hcurve],label=method_curve)
                for ax in axes: ax.set_ylabel('RMSE validation (unité cible)' if kind=='regression' else 'Exactitude validation')
                axes[0].set_xlabel('Époque'); axes[1].set_xlabel('Secondes'); axes[0].legend(fontsize=6)
                fig.suptitle(prefix+' - graine 42'); fig.tight_layout(); fig.savefig(OUT/(prefix+'_metrique.png'),dpi=140); plt.close(fig)
            if kind=='regression' and xe.shape[1]==2:
                fig,axes=plt.subplots(1,2,figsize=(8,3))
                truth=ye*ys+ym; prediction=model.predict(p,xe)*ys+ym
                for ax,values,title in zip(axes,[truth,prediction],['Cible observée test','Prédictions test']):
                    sc=ax.scatter(xe[:,0],xe[:,1],c=values,vmin=min(truth.min(),prediction.min()),vmax=max(truth.max(),prediction.max()),cmap='viridis'); ax.set(title=title,xlabel='x1 standardisé',ylabel='x2 standardisé'); fig.colorbar(sc,ax=ax,label='Unité cible')
                fig.suptitle(prefix+' - '+method+' - graine 42'); fig.tight_layout(); fig.savefig(OUT/(prefix+'_fonction.png'),dpi=140); plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(12,3))
            pred=model.predict(p,xe)
            if kind=='regression':
                truth=ye*ys+ym; pred=pred*ys+ym
                axes[0].scatter(truth,pred,s=12); lim=[min(truth.min(),pred.min()),max(truth.max(),pred.max())]; axes[0].plot(lim,lim,'k--'); axes[0].set(xlabel='Réel',ylabel='Prédit',title='Test')
                axes[1].hist(truth-pred,bins=15); axes[1].set(xlabel='Résidu réel - prédit',ylabel='Effectif',title='Erreurs test')
                if xt.shape[1]==2:
                    axes[2].scatter(xe[:,0],xe[:,1],c=truth,cmap='viridis',s=30); axes[2].set(title='Cible test (couleur)',xlabel='x1 standardisé',ylabel='x2 standardisé')
                else: axes[2].imshow(np.corrcoef(xt.T),vmin=-1,vmax=1,cmap='coolwarm'); axes[2].set_title('Corrélations train')
            elif kind=='classification':
                labels=pred.argmax(1); cm=confusion_matrix(ye,labels)
                axes[0].imshow(cm,cmap='Blues'); axes[0].set(xlabel='Prédit',ylabel='Réel',title='Matrice test')
                for i in range(len(cm)):
                    for j in range(len(cm)): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7)
                f1=precision_recall_fscore_support(ye,labels,zero_division=0)[2]; axes[1].bar(np.arange(len(f1)),f1); axes[1].set(xlabel='Classe',ylabel='F1',ylim=(0,1))
                if xe.shape[1]==2:
                    u,v=np.meshgrid(np.linspace(-2,2,150),np.linspace(-2,2,150)); zz=model.predict(p,np.column_stack([u.ravel(),v.ravel()])).argmax(1).reshape(u.shape)
                    axes[2].contourf(u,v,zz,alpha=.25); axes[2].scatter(xe[:,0],xe[:,1],c=ye,s=10); axes[2].set_title('Frontière et classes réelles test')
                else: axes[2].scatter(xe[:,0],xe[:,1],c=ye,s=5); axes[2].set_title('Deux premières variables test')
            else:
                c=p.reshape(model.k,-1)
                axes[0].scatter(xe[:,0],xe[:,1],c=pred,s=10); axes[0].scatter(c[:,0],c[:,1],marker='X',c='red',s=80); axes[0].set_title('Groupes et centroïdes')
                axes[1].bar(np.arange(model.k),np.bincount(pred,minlength=model.k)); axes[1].set(xlabel='Cluster',ylabel='Effectif')
                axes[2].imshow(confusion_matrix(ye,pred),cmap='Blues'); axes[2].set(xlabel='Cluster (numéro arbitraire)',ylabel='Classe réelle',title='Composition après apprentissage')
                # Lloyd : référence distincte, sans lecture des étiquettes.
                for seed in SEEDS:
                    lmodel=make(seed); centers=lmodel.initial.reshape(lmodel.k,-1); start=time.perf_counter()
                    for _ in range(epochs):
                        groups=distances(xt,centers).argmin(1); new=centers.copy()
                        for k in range(lmodel.k):
                            if np.any(groups==k): new[k]=xt[groups==k].mean(0)
                        if np.max(abs(new-centers))<1e-8: break
                        centers=new
                    results.append(dict(jeu=name,exercice=kind,modele=modelname,methode='Lloyd',graine=seed,temps=time.perf_counter()-start,**evaluate(lmodel,centers.ravel(),xe,ye)))
            fig.suptitle(prefix+' - '+method+' - graine 42'); fig.tight_layout(); fig.savefig(OUT/(prefix+'_details.png'),dpi=140); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=150); args=parser.parse_args()
    results,checks=[],[]
    paths=sorted((ROOT/'donnees').glob('*.csv'))+sorted((ROOT/'donnees').glob('*.npz'))
    for path in paths:
        print('Traitement :',path.name,flush=True); run_dataset(path,8 if path.suffix=='.npz' else args.epochs,results,checks)
        save_csv(OUT/'resultats_bruts.csv',results); save_csv(OUT/'gradients.csv',checks)
    summary=[]
    for key in dict.fromkeys((r['jeu'],r['modele'],r['methode']) for r in results):
        group=[r for r in results if (r['jeu'],r['modele'],r['methode'])==key]
        row=dict(zip(['jeu','modele','methode'],key)); row['graines']=len(group)
        for column in group[0]:
            if column in ['graine'] or column in row: continue
            values=[r[column] for r in group if isinstance(r.get(column),(int,float,np.number))]
            if values: row[column+'_moyenne']=float(np.mean(values)); row[column+'_ecart_type']=float(np.std(values))
        summary.append(row)
    save_csv(OUT/'resume.csv',summary)
    (OUT/'versions.json').write_text(json.dumps(dict(numpy=np.__version__,matplotlib=matplotlib.__version__,epochs=args.epochs,graines=SEEDS),indent=2))
    print('Terminé. Résultats dans',OUT)

if __name__=='__main__': main()
