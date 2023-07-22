"""
İki grup ortalamalarının arasında karşılaştırma yapılmak istenildiğinde kullanılır.
A: kontrol
B: deney grubudur.

Hipotez: işimizi şansa bırakmadan istatistiksel verilere dayanarak sonuca ulaşmak.

H0: M1 = M2 (yokluk hipotezidir. İki grup ortalamasında istatistiksel olarak anlamlı fark yoktur, eşittir.)
H1: M1 != M2

p<0.05 H0 RED
p>0.05 H0 REDDEDİLEMEZ

Bağımsız örneklem t testinin 2 ayrı varsayımları vardır:
1) 2 grubun da ayrı ayrı normal dağılması gerekmektedir.
2) 2 grubun varyans homojenliği varsayımıdır.
İki grubun dağılımlarının birbirine benzer olup olmamasıdır.


1. Hipotez kur
2. Varsayımları incele (Gerekirse keşifçi veri analizi + veri ön işleme yap)
3. p-value bakarak yorum yap.

Eğer varsayımlar sağlanıyorsa parametrik test
Varsayımlar sağlanmıyorsa non-parametrik test

1. Hipotez kur
2. Varsayım Kontrolü
3. Hipotezlerin Uygulanması
    3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test) yapılır.
    3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test) yapılır.
4. p-value bakarak yorum yap.

Not:
- Normallik varsayımları sağlanmıyorsa direkt mannwhitneyu testi (non-parametrik test) yapılır.
- Normal dağılım sağlanıyor, varyans homojenliği sağlanmıyorsa yine bağımsız iki örneklem t test(parametrik test)
kullanıyoruz ama varyanslar sağlanmıyor bilgin olsun diyoruz. Bağımsız iki örneklem t testine
argüman girilir.
- Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.
"""

# Uygulama: Sigara içenler ile içmeyenlerin hesap ortalamaları arasında istatistiksel
# anlamda fark var mı?

import seaborn as sns
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
pd.set_option("display.max.columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float.format", lambda x: "%.5f" % x)

df_ = sns.load_dataset("tips")
df = df_.copy()
df.head()

#Veri ön işleme
"""
Öncelikle sigara içenlerin ortalama bahşiş tutarına bakalım:
Aşağıda da görüldüğü üzere matematiksel olarak fark var gibi gözüküyor amaa
gerçekten fark var mı? bu fark istatistiksel olarak da var mı?
Burda bir fark görüyor olabilirim ama bu şans eseri de ortaya çıkmış olabilir.
İstatistiki olarak da test etmeliyiz.
"""
df.groupby("smoker").agg({"tip":"mean"})

# 1. Hipotez kurmak: Amaç işimizi şansa bırakmamaktır.
# H0: M1 = M2 (Yokluk Hipotezi)
# Sigara içnelerin ve sigara içmeyenlerin hesap ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.
# H1: M1 != M2
# Sigara içenler ile sigara içmeyenlerin hesap ortalamalrı arasında istatistiksel olarak anlamlı bir fark vardır.

# 2. Varsayımların Kontrolü:
# 2.1. Normallik Varsayımı
# (shapiro(): shapiro testi bir değişkenin dağılımının normal olup olmadığını test eder.)
# 2.2. Varyans Homojenliği
# (levene(): levene testi iki farklı grubun varyans homojenliğinin olup olmadığını test eder.)

# 2.1. Normallik Varsayımı
# H0: Normal dağılım varsayımları sağlanmaktadır.
# H1: Normal dağılım varsayımları sağlanmamaktadır.

#shapiro der ki bana ilgili grubu ve ilgili değişkeni ver.
#1. grup (sigara içenler grubu) için normallik dağılım testi:
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0002 < 0.05 olduğu için H0 RED
# 0.05 standartlaştırılmış tabloların köşe noktalarına denk geliyor olacak ve bu da
# şansa yer bırakmayacak şekilde test etme imkanı sağlayacaktır.

#2. grup (sigara içmeyenlerin grubu) için normallik dağılım testi:
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0000 < 0.05 olduğu için H0 RED

# HO RED olduğu için normallik dağılım varsayımı sağlanmamaktadır.
# Bundan dolayı non-parametrik bir test kullanmamız gerekir. (mannwhitneyu testi)

#Biz normallik dağılım varsayımı sağlanmış gibi davranalım:
# 2.2. Varyans Homojenliği

# H0: Varsayım Homojenliği sağlanmaktadır.
# H1: Varsayım Homojenliği sağlanmamaktadır.

# Varyans Homojenliği Testi:
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.0452 < 0.05 olduğu için H0 RED

# iki varsayım da sağlanmaz!
# Ama biz sağlanmış gibi davranıp ona göre test yapalım:

# 3. Hipotezlerin Uygulanması
#   3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test) yapılır.
#   3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test) yapılır.

# 3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test) yapılır.
# ttest_ind() : t testi der ki:
# - Eğer normallik varsayımları sağlanıyorsa beni kullanabilirsin.
# - Normallik varsayımları sağlanıyor ve varyans homojenliği sağlanıyorsa beni kullanabilirsin.
# - Normallik varsayımı sağlanıyor ama varyans homojenliği sağlanmıyorsa da beni kullanabilirsin.
# Ama sadece ttest_ind(equel_var=False)
# (Yani varyanslar eşit mi? True ise evet, false ise hayır, dolayısıyla varyans homojenliği sağlanmıyorsa equel_var=False olmalıdır!
# Böylece arka tarafta xxx testini yapar.)
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"], equal_var=True)
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#pvalue = 0.1820 > 0.05 olduğu için

#4. p-value bakarak yorum yap.
# p < 0.05 H0 RED
# P > 0.05 H0 REDDEDİLEMEZ
#pvalue = 0.1820 > 0.05 olduğu için H0 REDDEDİLEMEZ,
# Yani istatistiksel olarak anlamlı bir fark yoktur!


# 3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test) yapılır.
# mannwhitneyu testi ortalama/medyan kıyaslama testidir.
# İki ayrı grup girilir ve bunlar arasında anlamlı bir fark var mı kontrol eder.

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat= %.4f, p-value = %.4f" % (test_stat, pvalue))

#pvalue = 0.3413 > 0.05 H0 REDDEDİLEMEZ analamına gelir.
#Dolayısıyla iki grubun ortalamaları arasında anlamlı bir fark yoktur deriz!
#H0' ı ya red ederiz ya da reddedemeyiz. H1' i kabul etme ya da red etmek gibi bir durum yoktur!