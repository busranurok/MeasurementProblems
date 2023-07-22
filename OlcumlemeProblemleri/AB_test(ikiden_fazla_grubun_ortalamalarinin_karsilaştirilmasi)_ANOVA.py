# İkiden fazla grubun ortalamaları arasında karşılaştırma gerçekleştirmek istersek ANOVA testi yaparız.
# HO: M1 = M2 = M3
# H1: Eşit değillerdir.

# Uygulama: haftanın günleri arasında ortalama total_bill farkı var mı?

import seaborn as sns
import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

df_ = sns.load_dataset("tips")
df = df_.copy()

df.head()

#Günlerin ortalamaları açısından en azından fark var mı diye kontrol edelim:
df.groupby("day").agg({"total_bill":"mean"})

"""
      total_bill
day             
Thur   17.682742
Fri    17.151579
Sat    20.441379
Sun    21.410000

hafta içi ortalamaları birbirine yakın, hafta sonu ortalamaları birbirine yakın.
hafta içi ve hafta sonu arasında bir fark olduğu matematiksel olarak gözüküyor.
ama bu farklılık şans eseri çıkmış olabilir.
"""

# 1. Hipotezi Kur:
# H0: Günlerin total_bill' lerinin ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.
# H1: Günlerin total_bill' lerinin ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.

# 2. Varsayımların Kontrolü:
# 2.1. Normallik Varsayımı
# 2.2. Varyans Homojenliği Varsayımı

# Varsayımlar sağlanıyorsa one way anova testi
# Varsayımlar sağlanmıyorsa kruskal testi (non- parametrik ikiden fazla grubun ortalamalarının karşılaştırılması)yapılır.


# 2.1. Normallik Varsayımı Kontrolü:
# H0: Normal dağılım varsayımı sağlanmaktadır.
for group in list(df["day"].unique()):
    #shapiro[0] = ttest_stat, shapiro[1] = pvalue
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value: %.4f" % pvalue)

"""
Sun p-value: 0.0036
Sat p-value: 0.0000
Thur p-value: 0.0000
Fri p-value: 0.0409

hepsi için pvalue < 0.05 olduğu için H0 RED. Bu yüzden Normallik varsayımı sağlanmaz.
"""

# 2.2. Varyans Homojenliği Varsayımı
#H0: Varyansların homojenliği varsayımını sağlamaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

"""
p-value = 0.5741 > 0.05 olduğu için H0 REDDEDİLEMEZ. Varyans homojenliği varsayımı sağlar. 
Fakat zaten normallik varsayımı sağlanmadığı için k
"""

#Varsayımlar sağlanmış gibi düşünelim:
f_oneway(df.loc[df["day"] == "Sun", "total_bill"],
                df.loc[df["day"] == "Sat", "total_bill"],
                df.loc[df["day"] == "Thur", "total_bill"],
                df.loc[df["day"] == "Fri", "total_bill"])

#Varsayımlar sağlanmadığı için, non-parametrik test(kruskalctesti) yapılır:
kruskal(df.loc[df["day"] == "Sun", "total_bill"],
                df.loc[df["day"] == "Sat", "total_bill"],
                df.loc[df["day"] == "Thur", "total_bill"],
                df.loc[df["day"] == "Fri", "total_bill"])

#pvalue = 0.015433008201042065 < 0.05 H0 RED.
# Görüyoruz ki bu gruplar arasında anlamlı bir fark vardır.
# Peki bu fark hangisinden kaynaklıdır?
# Bunun için biz çoklu karşılaştırma yapacağız:
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

#İkili karşılaştırma yapıldığında anlamsal fark yoktur.
#İyi de ben anlamlı fark bulmuştum.
# p_value(alfa değerini) değiştiririm ya da fark yokmuş gibi düşünülebilir.
# pvalue = 0.01 or 0.10, istatistiksel olarak ne kadar tolerans gösterdiğini gösteriyor.
#anova da fark var dedirten gün pazar çıkar.