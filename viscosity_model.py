"""СРС 3. Прогнозирование вязкости ЛКМ (синтетические учебные данные)."""
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
N = 600
df = pd.DataFrame({
    "T": rng.uniform(20, 60, N),      # °C
    "n": rng.uniform(200, 1200, N),   # об/мин
    "t": rng.uniform(10, 90, N),      # мин
    "C": rng.uniform(30, 60, N),      # % пигмент+наполнитель
})
# нелинейная «физика»: вязкость растёт с C, падает с T и n (сдвиговое разжижение), стабилизируется во времени
df["mu"] = (120*np.exp(0.045*(df.C-30))*np.exp(-0.022*(df["T"]-20))
            *(df.n/500)**-0.25*(1+0.6*np.exp(-df.t/25)) + rng.normal(0, 8, N))
print(df.describe().round(1).T[["mean","std","min","max"]])
print(df.corr().round(2)["mu"])

X, y = df[["T","n","t","C"]], df["mu"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
res = {}
for name, m in [("Линейная регрессия", LinearRegression()),
                ("Случайный лес", RandomForestRegressor(300, random_state=1))]:
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res[name] = dict(MAE=mean_absolute_error(yte,p), RMSE=mean_squared_error(yte,p)**.5,
                     MAPE=np.mean(np.abs((yte-p)/yte))*100, R2=r2_score(yte,p))
    if name == "Случайный лес":
        rf, prf = m, p
        imp = dict(zip(X.columns, m.feature_importances_.round(3)))
print(pd.DataFrame(res).T.round(2)); print("importance", imp)

# Сравнение управления: цель mu=150±15 мПа·с; 400 партий со случайным отклонением сырья
def plant(T,n,t,C): return 120*np.exp(0.045*(C-30))*np.exp(-0.022*(T-20))*(n/500)**-0.25*(1+0.6*np.exp(-t/25))
B=400; C0=rng.normal(45,4,B).clip(32,58); ok_t=ok_i=0; E_t=E_i=0; ts=[]
for c in C0:
    # традиционное: фиксированный режим
    T,n,t=40,700,45
    m=plant(T,n,t,c)+rng.normal(0,8); ok_t+=abs(m-150)<=15; E_t+=n**3*t*1e-9
    # интеллектуальное: подбор n и T по прогнозу РЛ в допустимых пределах
    best=None
    for T2 in (35,40,45,50):
        for n2 in range(400,1101,100):
            pr=rf.predict(pd.DataFrame([[T2,n2,45,c]],columns=X.columns))[0]
            cost=abs(pr-150)+1e-9*n2**3*45*0.5
            if best is None or cost<best[0]: best=(cost,T2,n2)
    _,T2,n2=best; m=plant(T2,n2,45,c)+rng.normal(0,8); ok_i+=abs(m-150)<=15; E_i+=n2**3*45*1e-9
print("traditional ok %.1f%%, E %.2f"%(ok_t/B*100,E_t/B)); print("intelligent ok %.1f%%, E %.2f"%(ok_i/B*100,E_i/B))

# SVG scatter факт/прогноз
lo,hi=float(min(yte.min(),prf.min())),float(max(yte.max(),prf.max()))
sc=lambda v:30+(v-lo)/(hi-lo)*340
pts="".join(f'<circle cx="{sc(a):.1f}" cy="{400-sc(b)+30:.1f}" r="3"/>' for a,b in zip(yte,prf))
open("data/scatter.svg","w").write(pts)
json.dump({k:{a:round(b,2) for a,b in v.items()} for k,v in res.items()},open("data/res.json","w"))
df.to_csv("data/paint_viscosity_synthetic.csv",index=False)
