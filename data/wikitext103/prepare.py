import os
import pickle
import numpy as np

train_file_path = os.path.join(os.path.dirname(__file__), 'wiki.train.tokens')
validation_file_path = os.path.join(os.path.dirname(__file__), 'wiki.valid.tokens')

with open(train_file_path, 'r', encoding='utf-8') as f:
    train_data = f.read()
with open(validation_file_path, 'r', encoding='utf-8') as f:
    val_data = f.read()

data = train_data + val_data

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

"""
all the unique characters: 
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¥§©«­®¯°±²³´µ¶·¹º»¼½¾¿ÀÁÂÄÅÆÇÈÉÌÍÎÑÓÔÖ×ØÚÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀāăąĆćČčĎĐđēėęěğġĦħĩĪīİıķĽľŁłńņňŋŌōŏőŒœŘřŚśŞşŠšŢţťũūŭůųŵŷŹźŻżŽžƏƒơưǂǎǐǒǔǫȘșȚțɐɑɒɓɔɕɖɗəɛɟɡɢɣɦɧɨɪɬɯɲɴɵɸɻɾʀʁʂʃʇʈʊʋʌʍʎʒʔʕʘʝʟʰʲʷʻʼʾʿˀˁˈˌː˔˘˚˞ˠˤ˥˦˧˨˩̧̝̞̟̠̣̤̥̩̪̯̰̱̲̺͍̀́̂̃̄̆̈̊̌̍̐̽̚͘͜͡ΑΒΓΔΕΗΘΙΚΛΜΝΞΟΠΣΤΦΧΨΩάέήίαβγδεζηθικλμνξοπρςστυφχψωόύώϕЈАБВГДЕЗИЙКМНОПРСТЧШЯабвгдежзийклмнопрстухцчшъыьэюяёіјљўԿՀՄՍաբեթիկհղյնշոպսվտրցւքֵֶּ֤֫אבגדוחיכלםמןנסעצרשת،ءأإابةتثجحخدرزسشصطعغفقكلمنهويیंअआउकगटडतदनपबभमयरलशषसह़ािीुेो्।॥কতনবমল়াি্கனள்్್്්กขงจชฐณตทนปพมยรลวะัาิีู฿เแ็่้์་།ငဆနရ္်გდვზიკორსუცძწხჯ჻፡ᛃᛋᛟតនពមរសហីុោះ់្᷉ḍḑḤḥḩḷḻṃṅṇṉṛṟṢṣṭṯạảấầẩẫậắằẵặẹếềểễệịỌọỏốồổỗộớờởợụủứừửữựỳỵỹἀἄἈἐἝἡἨἰἱἵἶἸὁὌὐὑὡὰὲὴὶὸὺὼᾶῆῖῦῬῶῷ​‌‍‐‑‒–—―‘’‚“”„†‡•…‰′″※‼‿⁄⁊⁡⁺₁₂₃₡₣₤₦₨₩₫€₱₹℃ℓ№™⅓⅔⅛⅜⅝⅞←↑→↓↔↗↦↪⇄⇌∀∂∆∈∑−∕∖∗∘√∝∞∩∪∴∼≈≠≡≢≤≥≪⊂⊕⊗⊙⊥⋅⋯⌊⌋①②④█▲★☆☉♀♠♣♥♦♩♪♭♮♯⟦⟧⟨⟩⩽ⴷ、。〈〉「」『』〜あいぅうおかがきくぐこさしじすずせたちつてとどなのはばふへぽまみめよらりるんァアィイウェエォオカガキギクグケコゴサザシジスズセゼソタダチッツテデトドナニノハバパヒビフブプベペボポマミムメモャヤュユョラリルレロワンヴ・ー一三上下不丑世个久之乙二五亦京人介仙令伝住佛作併侠個催像光内写冴净凪分列刘初判別前剛剣劇動勝北匹千卒南博厘双台号同名君呼哈四回国國土坂場境士外大天太女姉婷子字季孩守宋宗寔寺小少尾巨廷式弘张張彼後心思愛憲戦房所撃攻新方日旦早星春時書月朝木本李杜束条東松枚條植椎楽様機正武歩死殻氏民気氣水氷氾法活流浅海溜瀬火無版特王瓦生用田町界畫皮真礮神秋積空章箇節米約紋綾緑編群羽耕聖肖膺芝花草莊蛍街裁装西見語説議词谷路軍転輝逆進道遠邪部野量金銀銖錬隊隴隻雄集雪零靈青韓風鬼魂魄魔鼓거루마막말사인전지짓투하ﬁ﻿！＆（），－：＝？～｢･￥
vocab size: 1,250
train has 538,360,726 tokens
val has 1,140,678 tokens
"""