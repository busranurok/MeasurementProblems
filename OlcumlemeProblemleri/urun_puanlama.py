"""
Olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlama:
Rating Products

-Average
-Time Based Weighted Average
-User Based Weighted Averaged
-Weighted Rating
-Bayesian Average Rating Score(Olasılıksal ortalama, bu kursların puanlarını bir miktar kırpıp daha aşağıda gösterebilir.)
"""

#Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float.format", lambda x: "%.5f" %x)

"""
(50+ Saat) Python A-Z: Veri Bilimi ve Machine Learning
Puan: 4.8 (4.764925)
Toplam Puan: 4611
Puan Yüzdeleri: 75, 20, 4, 1, <1
Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6
Amacımız: Bu kursa verilen puanların puanını hesaplamak
"""

df_ = pd.read_csv("dataset/course_reviews.csv")
df = df_.copy()
"""
Rating: Bu kursa puan veren kişileri ifade eder
Timestamp: Kursa hangi tarihte yorum verdiğini gösteriyor
Enrolled: Kişinin hangi tarihte üye olduğunu ifade eder
Progress: Kursun %kaçını izlediği bilgisini tutar
Questions Asked: Ne kadar soru sorduğu bilgisini tutar
Questions Answered: Yanıt aldığı soruların sayısını ifade eder
"""
#Veri setini tanımak için
df.head()
# Kaç tane değerlendirme mevcut: (4323, 6)
df.shape

# rating(puanların) dağılımı, hangi puandan kaçar tane olduğu bilgisidir.
df["Rating"].value_counts()

# Sorulan sorular değişkeni hakkında bilgi alalım:
df["Questions Asked"].value_counts()

# Sorulan soru kırılımında verilen ortalama puan:
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating":"mean"})

#Average
"""
Direkt böyle bir ortalama kullanıldığında ürünle ilgili birçok trendi kaçırıyor olabiliriz.
Mesela müşteri memnuniyeti, kargo hizmetlerinin, paketleme hizmetleri, iade hizmetlerinin ne durumda olduğu bilgisi.
"""
df["Rating"].mean()

#Puan zamanlarına Göre Ağırlıklı Ortalama alınmalıdır
#Time-Based Weighted Average
df.head()
df.info()

#Timestamp değişkeninin veri tipi object, onu datetime a çevirmeliyiz:
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

#Bu tarih bu veri setindeki max tarih, 1 değerini gördüğümüzde 1 gün önce yprum yapılmış anlamına gelecek
current_date = pd.to_datetime("2021-02-10 0:0:0")

#Bugünkü tarihten yorumu yaptığı tarih çıkarıldı ve gün cinsine çevrildi.
df["days"] = (current_date - df["Timestamp"]).dt.days

df.head()

#Bu veri setinde son 30 günde yapılan yorumlardan kaç tane vardır?:
df.loc[df["days"] <= 30].count()

#30 dan küçük:
df.loc[df["days"] <= 30, "Rating"].mean()

#Son 30 günde yapılan yorumların ortalamasını almak istersek, sadece Rating' ler gelsin:
df.loc[df["days"] <= 30, "Rating"].mean()

#30 dan büyük, 90 dan küçük eşit olanların da ortalamasını alın:
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

#90 dan büyük 180 den küçük eşit olsun, ortalamasını alalım:
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

#180 den büyükse:
df.loc[(df["days"]) > 180].mean()

# Geçmişe gittikçe rating düşüyor. Bu da şu anlama geliyor: Son zamanlarda bu kursun puanında bir artış mevcut.

#Amacımız farklı zamanlara farklı ağırlıklarda odklanmaktı, böylece zamanın etkisini ağırlıklara yansıtabiliriz.
#Tarihin önemine göre ağırlık verilecektir: En önemlisine, en güncel olanı en yüksek puan, burada yakın tarih bizim için önemli olduğu için ona daha yüksek puan verdik:
#Ağırlıkların toplamı 100 olmalı.
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"]) > 180, "Rating"].mean() * 22/100

#Fonksiyonlaştıralım:
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1/100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2/100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3/100 + \
           dataframe.loc[(dataframe["days"]) > 180, "Rating"].mean() * w4/100


time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

#Peki her kullanıcının verdiği puan aynı etkiyi mi göstermeli, kursun tamamını izleyen kişi ile sadece giriş kısmını izleyen eşit mi olmalı?
#User-Based(User Quality) Weighted Average (Kullanıcı Temelli Ağırlıklı Ortalama)
#Yeni ürün alıp hemen 5 puan vermiş kişi ile birçok ürün satın alıp 5 puan vermiş kişi sosyal ispat ta farklı anlama gelir.
df.head()

#Görüldüğü üzere kursun izlenmesi ile verilen puan arasında bir ilişki mevcuttur:
df.groupby("Progress").agg({"Rating": "mean"})

#O halde, %10' undan az izleyenler şu, %75 bu vb.:
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22/100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26/100 + \
    df.loc[(df["Progress"]) > 75, "Rating"].mean() * 28/100

#Bunları yapmaktaki amacımız ortalamayı hassaslaştırmaktır Kursu en çok izleyen kursu daha iyi tanır.

#Fonksiyonlaştıralım:
def used_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1/100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * w2/100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * w3/100 + \
           dataframe.loc[(dataframe["Progress"]) > 75, "Rating"].mean() * w4/100

#Kullanıcıların satın alma sayısı, puan sayısı, yorum sayısı
used_based_weighted_average(df, 20, 24, 26, 30)


#Daha da hassaslaştırmak için, ağırlıklı derecelendirme(Weighted Rating) işlemi gerçekleştiririz:
#Time-based ve user-based de yaptığımız işlemleri tek bir fonksiyon kullanarak hesaplarız:
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + used_based_weighted_average(dataframe) * user_w/100


course_weighted_rating(df, time_w=40, user_w=60)

"""
Elimizde kullanıcı puanları var bunların direkt ortalamalarını almamız yetmiyor. Ortalamayı hassaslaştırmamız gerekiyor.
Bunun için ilk olarak zamana göre ortalama ağırlıklarını buluruz. Sonrasında kullanıcı kalitesine göre ortalamaların ağırlıklarını buluruz.
Ve ikisini tek bir fonksiyonda toplarız. Böylece daha hassas, daha güven verici, daha fazla faktörü bir arada bulunduran
ortalama hesabını gerçekleştirmiş olduk.
"""




