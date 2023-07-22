"""
Shorting Product(Ürün sıralaması)
- Sorting by Rating
- Sorting by Comment Count or Purchase Count
- Sorting by Rating, Comment and Purchase Count
- Sorthing by Bayesian Average Rating Score (Sorting Product with 5 Star Rated)
- Hybrid Sorting: BAR_SCORE + Diğer faktörler

Mesela: Bir mülakat sürecimiz olsun, bu süreçte yabancı dil bilbisi, mezuniyet ve çalışma süresi göz önünde bulunduruluyor.
Bu 3 faktörden hangisine göre sıralama yapılması gerekir?
Burada ağırlıklı puan verip sonuca varabiliriz.

Kursun puanını mı, yorum sayısını mı yoksa satın alınma sayısını mı dikkate almalıyız?
"""

#Kurs sıralaması yapalım:
import pandas as pd
#Standartlaştırmak için kullanırız
from sklearn.preprocessing import MinMaxScaler
#Bayesian_average_rating fonksiyonunu kullanabilmek için import edildi
import math
import scipy.stats as st
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" %x)

df_ = pd.read_csv("dataset/product_sorting.csv")
df = df_.copy()
df.head()
df.shape

"""
course_name: Kurs ismi
instructor_name: Eğitmen isimleri
purchase_count: Satın alma sayıları
rating: kursun ortalama puanı
comment_count: kursun aldığı yorum sayısı
5_point: Yorumlar ile verilen puanları(yorumlu ve yorumsuz puan) veren kişi sayısı
"""

#Elimizde ürünler mevcut, bunu sıralamak istiyoruz. İstatistiksel bir şekilde sıralama yapacağız:
df.sort_values("rating", ascending=False).head(10)

"""
Sadece rating' e göre sıralama yaptığımızda, satın alma sayısı ve yorum sayısı rating' in altında ezilmiş olur. 
Satın alma sayısı ve yorum sayısını da göz önünde bulundurmak gerekir.
"""

df.sort_values("purchase_count", ascending=False).head()
"""
Yorumu az olanlar ve düşük puan alanlar da yukarıya geldi. Bu yüzden tek başına bu metriği de kullanmak mantıklı değil.
"""

df.sort_values("commment_count", ascending=False).head()
"""
Bu metrik de tek başına sıralama için yeterli değildir. Bu yüzden bütün metrikleri göz önünde bulundurarak standartlaştırma işlemi yapıp
sıralamayı gerçekleştireceğiz. (Yorum sayısı, satın alma sayısı, puan sayısı)
"""

#Shorting by Rating, Comment and Purchase
#Değerleri direkt çarpmak doğru değildir çünkü ölçekleri birbirinden farklıdır. Ölçeği büyük olan küçüğü ezer.
# Bu yüzden hepsini aynı ölçeğe getirmek çözüm olacaktır.
df["purchase_count_scale"] = MinMaxScaler(feature_range=(1, 5)).\
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.head()

df.describe().T


df["comment_count_scale"] = MinMaxScaler(feature_range=(1, 5)).\
    fit(df[["commment_count"]]).\
    transform(df[["commment_count"]])

df.head()

"""
Artık bu değişkenlerin hepsi aynı cinsten, ağırlıklı ortalamalarını da ortalamalarını da alabiliriz. Hepsini çarpıp sıralayabiliriz.
Ama daha da hassaslaştırmak adına ağırlıklarıı verebiliriz.
"""

#Hesapladığı şey rating değil rating skorlarıdır. Belirli bir şeylerin bir araya gelerek oluşturduğu şey olduğu için skordur.
#Benim için en önemli şey puandır. Sonrasında yorumdur.
(df["purchase_count_scale"] * 26/100 +
 df["commment_count"] * 32/100 +
 df["rating"] * 42/100)


def weighted_sorting_score(dataframe, w1=26, w2=32, w3=42):
    return (dataframe["purchase_count_scale"] * w1/100 +
            dataframe["commment_count"] * w2/100 +
            dataframe["rating"] * w3/100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.head()

df.drop("purchase_count_scaled", axis=1, inplace=True)

# Sosyal ispat gördüğüm 3 faktörü aynı anda kullandığımızda kurs sıralaması değişti. Artık buna güvenebilirim.
# wss
df.sort_values("weighted_sorting_score", ascending=False).head()

#Alakasız olanları çıkartarak sıralayalım, kurs isminde Veri Bilimi olanları getireceğiz:
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head()

"""
Elimizde birden fazla faktör olduğunda, bu faktörler ile ilgili bir eş zamanlılığı göz önünde bulundurayım ve sıralayım ihtiyacı 
doğduğunda, önce faktörleri aynı standarda getiriyoruz, sonra ister ortalaması(eşit ağırlık) ister ağırlıklı ortalamasını alıp sıralamasını alıyoruz..
"""

"""
Bayesian Average Rating Score (Ürün puanı olarak da değerlendirilebilir)

Daha önce rating hesaplamamızı hassaslaştırmıştık. Kullanıcılara göre, zamana göre dokunuşlar yapmıştık.
Sonrasında bir sıralama yapmak istediğimizde bu rating' in tek başına yeterli olamayacağı görüşüyle bazı diğer faktörleri de göz önünde bulundurarak
(purchase_count, comment_count) kayda değer sıralama gerçekleştirdik.
Acaba ratingleri başka açılardan hassaslaştırabilir miyiz? Ya da sadece rating' e odaklanarak bir sıralama yapabilir miyiz?

* Bu konu 5 yıldızlı sistemlerde ürün sıralama
* ya da 5 yıldızın dağılımına göre ürün sıralama
Odağımız 5_point, 4_point vb' nin dağılımını kullanarak olasılıksal bir ortalama hesaplayacağız.

Puan dağılımları üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar. Bu puanların dağılımı üzerinden bir ortalama hesaplayacağız.
n: girilecek olan yıldızların, gözlemleri. Yani 1 yıldızdan kaç tane var, 2 yıldızdan kaç tane var...
confidence: hesaplanacak olan z tablo değerine ilişkin bir değer elde edebilmek için girilmiş değerdir.
"""
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

#bar_average_rating, bar_shorting_score, bar_rating şeklinde de isimlendirilebilir. Var olan ratingleri biraz aşağıda gösterdiği için score
# demeyi tercih ederiz.

#lambda fonksiyonu seçme işlemi gerçekleştirir.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head()
#Bu yöntem bize rating lere göre sıralama yaptırır ama yin epurchase_count, comment_count' lar gözden kaçar.
#Tek odağımız verilecek puanlar ise, buna göre sıralama yapılacak ise bu yeterlidir.
df.sort_values("bar_score", ascending=False).head()

#5 ve 1 index e sahip mi, bu indexten olanları getir.
df[df["course_name"].index.isin([5-1])].sort_values("bar_score", ascending=False).head()

#Ratingler için olasılıksal çok güzel çözümümüz var ama yine bir problem var. Daha fazla kişi yorum vermiş ama yıldızı daha az ürün
#bizim için daha ön planda olması gerekirken daha aşağıda gelir. Bunu çözmek için hybrid yaklaşım sergilememiz gerekir:
#Bütün iş yerlerinde diğer faktörler çok önemlidir. Mesela yorum sayısı sosyal ispattır(social proof).
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

    wss_score = weighted_sorting_score(dataframe)

    return  bar_score * bar_w/100 + wss_score * wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.head()

#bar_score potansiyeli yüksek ama henüz yeterli social proof alamamış ürünleri de yukarı çıkarır.

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head()


"""
Not: 
1. İş bilgisi açısından önemli olan faktörler göz önünde bulundurulmalıdır.
2. Eğer birden fazla faktör varsa bu faktörler standartlaştırılmalı daha sonra etkilerinini farkı varsa bu ağırlık ile ifade edilmelidir.
3. İstatistiksel bazı faktörleri bulsak bunlar güvenilir dahi olsa bu yöntemleri tek başına kullanmak yerine iş bilgisi ile 
harmanlanacak şekilde kullanmak olacaktır.
harmanlanacak şekilde kullanmak olacaktır.
"""