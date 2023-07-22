"""
Biz düşük ya da yüksek puan ile ilgilenmiyoruz, pazar yeri olarak kullanıcılara en doğru social proof u göstermeye çalışıyoruz.
Sıralamayı etkileyen şey: user quality ya da ilgili iş birimini ifade edecek scoreler olabilir.
"""
import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

#Up-Down Different Score = (up rating) - (down rating)

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

"""
Farklılıktan dolayı 2 gibi gözükse de oranlardan dolayo 1 dir. Bu yetersiz kalır.
Başka yöntemler denenmelidir.
"""

#Average Rating = Score = (up rating) / (all rating) (faydalı oranı)
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101
score_average_rating(2, 0)

score_average_rating(100, 1)
# az önceki oran problemini çözer fakat bu sefer de frekans bilgisini kaçırır.


#Wilson Lower Bound Score (Wilson Alt Sınır Puanı)
#İkili interaction' lar barındıran herhangi bir item, product ya da review' ı score' lama imkanı sağlar.
#Bernoulli(iki sonucu olan olayların gerçekleşme olasılığını hesaplar) parametresi p için bir güven aralığı hesaplar.
#Bu aralığın alt sınırını wlb score' si olarak kabul eder.

# 600-400 : 0.6 => yorumlar
# 0.5 0.7 => güven aralığı
# 0.5 alt sınır olrak belirlerim. Bunu referans alarak sıralama yapabilirim.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasındaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli' ye uygun hale getirilir.
    Bu beraberinde bazı problemleri de getirir. bu sebeple bayesian average rating yapmak gerekir.
    Parameters
    ----------
    up: int
         up count
    down: int
          down count
    confidence: float
                confidence (güven aralığı: çok yaygınca 0.95 kullanılır.)

    Returns
    -------
    wilson score: float

    """
    #p istatistiğinin yani oran istatistiğinin güven aralığı formülüdür.
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1- phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

#oranları göre işlem yaptı. Bu yüzden 1. daha önemli oldu bizim için.

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)

#Uygulama:

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})

#score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                            x["down"]),
                                                axis=1)

#score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"],
                                                                               x["down"]),
                                                  axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],
                                                                            x["down"]),
                                                axis=1)

