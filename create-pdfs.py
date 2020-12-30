import os

chapters = (
    ("odkrivanje-skupin", 7),
    ("metoda-voditeljev", 19),
    ("razvrscanje-besedil", 31),
    ("projekcije", 39),
    ("linearna-regresija", 51),
    ("regularizacija", 61),
    ("logisticna-regresija", 67),
    ("klasifikacijska-drevesa-in-gozdovi", 73),
    ("priporocanje", 83),
    ("povezovalna-pravila", 95),
    ("END", 101)
)


def pairs(x):
    for i in range(len(x)-1):
        yield x[i], x[i+1]

for a, b in pairs(chapters):
    cmd = "pdfjam zapiski.pdf %d-%d -o pdfs/%s.pdf" % (a[1], b[1]-1, a[0])
    os.system(cmd)
