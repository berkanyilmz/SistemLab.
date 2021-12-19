import matplotlib.pyplot as plt
import numpy as np
import pandas

df = pandas.read_csv('/home/berkan/Spyder/veriseti.txt', header=None) #veri setini programa dahil edildi
print(df)

giris = df.iloc[0:10, [1,2,3,4,5,6,7]].values #veri setinden giriş değerleri alındı
cikis = df.iloc[0:10, [8]].values #veri setinden çıkış değerleri alındı
print(giris)


plt.title('2D görünüm', fontsize=16)

plt.scatter(giris[:5, 0], giris[:5, 1], color='black', marker='o', label="ilk 5")  #İlk 5 veri grafikte gösterildi
plt.scatter(giris[5:10, 0], giris[5:10, 1], color='green', marker='x', label="Son 5") #Son 5 veri grafikte gösterildi

plt.legend(loc='upper left')

plt.show()

class Perceptron(object): 
    def __init__(self, ogrenme_orani=0.1, iter_sayisi=10):
        self.ogrenme_orani = ogrenme_orani
        self.iter_sayisi = iter_sayisi

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.hatalar = []
        
        for _ in range(self.iter_sayisi):
            hata = 0
            for xi, hedef in zip(X, y):
                degisim = self.ogrenme_orani * (hedef - self.tahmin(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                hata += int(degisim != 0.0)
            self.hatalar.append(hata)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0] #İki matrisi çarpar ve yeni ağırlıkları belirler

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)
    

siniflandirici = Perceptron(ogrenme_orani=0.1, iter_sayisi=10)
siniflandirici.ogren(giris, cikis)

plt.plot(range(1, len(siniflandirici.hatalar) + 1), siniflandirici.hatalar)
plt.xlabel('Deneme')
plt.ylabel('Hatalı tahmin sayısı')
plt.show()