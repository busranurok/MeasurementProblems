# İki grubun oranları arasında istatistiksel olarak anlamlı bir fark olup olmadığını kontrol ederiz.

#İki ilgili grup için de örnek sayısı 30 dan büyük olmalı

#Uygulama: titanic veri setindeki kadın ve erkeklerin hayatta kalma oranlarında istatistiksel olarak anlamlı bir fark var mı?

import seaborn as sns
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

df_ = sns.load_dataset("titanic")
df = df_.copy()

# H0: P1 = P2 (P1 - P2 = 0)
# (Yani titanic veri setinde kadın ve erkeklarin hayatta kalma oranları arasında
# istatistiksel olarak anlamlı bir fark yoktur.)
# H1: P1 != P2
# (Yani titanic veri setinde kadın ve erkeklarin hayatta kalma oranları arasında
# istatistiksel olarak anlamlı bir fark vardır.)

# Bu testi yapacağımız fonksiyon( propotions_ztest() ) her iki grup için başarı sayısını
# Aynı zamanda her iki grup için gözlem sayısını alır. Bunları ayrı array' e koyup gönder.

# Kadınların hayatta kalma oranı:
df.loc[df["sex"] == "female", "survived"].mean()
# 0.7420382165605095

# Erkeklerin hayatta kalma oranı:
df.loc[df["sex"] == "male", "survived"].mean()
# 0.18890814558058924

#Bariz bir şekilde fark var fakat test yapıp yapılmaması gerektiği tartışmalı.
#Kadınlar için başarı sayısı:
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
#Erkeler için başarı sayısı:
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

#shape[0] yapmamız bize gözlem sayısını ver demektir. shape bize (gözlem sayısı, değişkenler) verir.
test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0000 < 0.05 olduğu için H0 RED