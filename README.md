
# 1. はじめに
このリポジトリは、DL4USの最終課題として取り組んだプロジェクトの資産の一部を公開するリポジトリです。当該プロジェクトは、ディープラーニング技術の習得のために取り組んだものであり、結果の正当性については保証致しません。

# 2. リポジトリのファイル構成
- Readme.ipynb : 本ファイル
- Case_1.ipynb : Case 1の実行結果
- Case_2.ipynb : Case 2の実行結果
- Case_3.ipynb : Case 3の実行結果
- gan.ipynb    : GANを用いたコード(実装途中)

# 3. プロジェクトの概要
本プロジェクトは、特許出願の願書に添付される特許請求の範囲の記載について英語から日本語に翻訳するタスクを深層学習によって行うプロジェクトである。

# 4. データ
## 4.1 コーパス
特許の機械翻訳用のコーパスとしては、国立情報学研究所が提供している[NTCIR-10](http://research.nii.ac.jp/ntcir/permission/ntcir-10/perm-en-PatentMT.html)が知られている。しかしながら、大学に所属する研究者等でなければ提供を受けられないなど、一般には使用できない。そのため、本プロジェクトでは、国際特許出願の願書に添付された特許請求の範囲の和文と、その国際特許出願の欧州移行時における特許請求の範囲の英文とから、英文と和文の約37万組からなるコーパスを新たに生成した。本プロジェクトでは、使用できるメモリ量や計算速度の観点から、英文と和文がともに所定の長さ以下のデータのみを使用して学習を行なった。なお、生成したコーパスは、以下からダウンロード可能である。

Corpus URL: https://drive.google.com/open?id=14mfeBRioi9dkAk546Aez6uaJcZeLoNQd


## 4.2 単語の分散表現
本プロジェクトでは、単語の分散表現として学習済みのものを使用し、分散表現の学習は行わない。本プロジェクトでは、Facebookが提供しているfastTextによる学習済みの[英単語の分散表現](https://fasttext.cc/docs/en/english-vectors.html)および[日本語の単語の分散表現](https://fasttext.cc/docs/en/crawl-vectors.html)を使用する。分散表現の次元は、いずれも300である。

# 5. モデル
本プロジェクトでは、ベースとなるモデルとしてRNN Encoder-Decoderを使用し、Encoderの出力系列の各要素にアテンドするAttentionを組み込んでいる。Encoder側のLSTMはBidirectionalとし、AttentionにはSoft Attentionを使用している。kerasによる実装の一部を以下に記載する。コードの詳細は、case_3.ipynb内のRNNEncoderDecoderAttのクラス定義を参照されたい。


```python
def inference(self):
    from keras.models import Model
    from keras.layers import Input, Embedding, Dense, LSTM, concatenate, dot, add, Activation

    # Encoder
    ## Input Layer
    self._encoder_inputs = Input(shape=(self.dim_input_seq,))

    ## Embedding Layer
    encoded_seq_fwd = Embedding(self.dim_input_vocab,
                                self.dim_emb,
                                weights=[self.emb_matrix_input],
                                mask_zero=True,
                                trainable=False)(self._encoder_inputs) # (dim_seq_input,)->(dim_seq_input, dim_emb)

    encoded_seq_bwd = Embedding(self.dim_input_vocab,
                                self.dim_emb,
                                weights=[self.emb_matrix_input],
                                mask_zero=True,
                                trainable=False)(self._encoder_inputs) # (dim_seq_input,)->(dim_seq_input, dim_emb)

    ## LSTM Layer
    for i in range(self._num_encoder_bidirectional_layers):
        encoded_seq_fwd, *encoder_states_fwd = LSTM(self.dim_hid, return_sequences=True, return_state=True)(encoded_seq_fwd)  # (dim_seq_input, dim_emb)->(dim_seq_input, dim_hid)
        encoded_seq_bwd, *encoder_states_bwd = LSTM(self.dim_hid, return_sequences=True, return_state=True, go_backwards=True)(encoded_seq_bwd)

        self._encoder_states.append([add([encoder_states_fwd[j], encoder_states_bwd[j]]) for j in range(len(encoder_states_fwd))])

    self._encoded_seq = add([encoded_seq_fwd, encoded_seq_bwd])

    # Decoder
    ## Instance
    self._decoder_embedding = Embedding(self.dim_output_vocab,
                                        self.dim_emb,
                                        weights=[self.emb_matrix_output],
                                        trainable=False)

    for i in range(self._num_decoder_RNN_layers):
        self._decoder_lstm.append(LSTM(self.dim_hid, return_sequences=True, return_state=True))

    ## Input Layer
    decoder_inputs = Input(shape=(self.dim_output_seq,))

    ## Embedding Layer
    decoded_seq = self._decoder_embedding(decoder_inputs)  # (dim_seq_output,)->(dim_seq_output, dim_emb)

    ## LSTM Layer
    for i in range(self._num_decoder_RNN_layers):
        decoded_seq, _, _ = self._decoder_lstm[i](decoded_seq, initial_state=self._encoder_states[i]) # (dim_seq_output, dim_emb)->(dim_seq_output, dim_hid)

    # Attention
    ## Instance
    self._attention_score_dense = Dense(self.dim_hid)
    self._attention_dense = Dense(self.dim_att, activation='tanh')

    ## Attention
    score = self._attention_score_dense(decoded_seq)        # (dim_seq_output, dim_hid) -> (dim_seq_output, dim_hid)
    score = dot([score, self._encoded_seq], axes=(2,2))           # [(dim_seq_output, dim_hid), (dim_seq_input, dim_hid)] -> (dim_seq_output, dim_seq_input)
    attention = Activation('softmax')(score)                # (dim_seq_output, dim_seq_input) -> (dim_seq_output, dim_seq_input)

    ## Context
    context = dot([attention, self._encoded_seq], axes=(2,1))     # [(dim_seq_output, dim_seq_input), (dim_seq_input, dim_hid)] -> (dim_seq_output, dim_hid)
    concat = concatenate([context, decoded_seq], axis=2)    # [(dim_seq_output, dim_hid), (dim_seq_output, dim_hid)] -> (dim_seq_output, 2*dim_hid)
    attentional = self._attention_dense(concat)             # (dim_seq_output, 2*hid_dim) -> (dim_seq_output, dim_att)

    # Output Layer
    ## Instance
    self._output_dense = Dense(self.dim_output_vocab, activation='softmax')

    ## Output
    predictions = self._output_dense(attentional)  # (dim_seq_output, dim_att) -> (dim_seq_output, dim_vocab_output)

    return Model([self._encoder_inputs, decoder_inputs], predictions)
```

# 6. 数値実験
モデルとデータ量との組み合わせに応じて、下表に記載した3つのケースについて数値実験を行なった。

|     |Model|Encoder sequence length|Decoder sequence length|Data volume|
|:---:|:-----------------------------------------:|:-:|:-:|:---:|
|Case1|1 LSTM layer + soft attention              |45 |106|Small|
|Case2|2 Bidirectional LSTM layers+ soft attention|45 |106|Small|
|Case3|2 Bidirectional LSTM layers+soft attention |41 |47 |Large|

Small dataとLarge dataのデータ数の内訳は下表の通りである。

|     |Training|Validation|Test|
|:---:|:---:|:--:|:-:|
|Small|10313|2579|131|
|Large|28146|3128|316|

## 6.1 学習
上記3つのケースのそれぞれについて、損失関数の値および予測精度の推移を以下に示す。

- Case 1
![comment](Data/920/history_040.png)

- Case 2
![comment](Data/920/history_043.png)

- Case 3
![comment](Data/920/history_070.png)

学習に要した時間は下表の通りである。

|     |Epochs|Total Time|
|:---:|:---:|:--:|
|Case1|35|About 57 min|
|Case2|45|About 135 min|
|Case3|45|About 195 min|


## 6.2 テスト
テストは、平均BLEUスコアに基づいて行なった。テストデータに対する平均BLEUスコアの推移を以下に示す。なお、本ドキュメント末に、学習後のモデルによって生成された翻訳文のうち、BLEUスコア Top 5の文を記載している。

![comment](Data/920/bleu.png)

# 結言
本プロジェクトによって、知財実務において実用に耐えうる機械翻訳器を現実的なコストで構築できる可能性が見えてきた。今後は、より複雑なモデルかつより多くの学習データを使用して、より長い系列での実用に耐えうる機械翻訳器を実現することを目指す。  

また、学習済みモデルで生成した文の中には、例えば請求項の項番号の誤りなど、人間が見れば誤訳であることが明らかな箇所がいくつか見られた。これらの誤りは、学習データを増やすことよりも、GANを使用することによって効率的に改善できるのではないかと考える。  

今回は、特許請求の範囲のデータを使用したが、知的財産分野は、法制度の特徴を上手く利用すれば学習データを構築し易い分野だと考える。翻訳タスクを通じて知財分野に適した深層学習モデル構築のノウハウを集積しつつ、明細書等の自動生成にも取り組みたいと考える。その取り組みの過程で、発明の自動探索を目指したモデルの構築も模索したい。

# Appendix
## BLEU Top 5 sentences

### Case 3: 2 Bidirectional LSTM layers + Large data set

元の文: the complex according to claim 1  which is represented by the following formula :  
正解文: 下式で表される請求項１に記載の複合体。  
生成文: 下式で表される請求項１に記載の複合体。  
BLEU: 0.8954237688029468  

元の文: the wort according to claim 12  wherein the α acid content is from 0 to 0.03 ppm inclusive .  
正解文: α酸の含量が０～０．０３ｐｐｍである、請求項１２に記載の麦汁。  
生成文: α酸の含量が０～０．０５ｐｐｍである、請求項１２に記載の麦汁。  
BLEU: 0.8944271909999159

元の文: the lead-acid battery of claim 6  wherein the flake graphite has an average primary grain diameter of 100 µm or more .  
正解文: 前記鱗片状黒鉛は、平均一次粒子径が１００μｍ以上である請求項６に記載の鉛蓄電池。  
生成文: 前記鱗片状黒鉛は、平均一次粒子径が１００μｍ以上である請求項６に記載の鉛蓄電池。  
BLEU: 0.8876027248484174  

元の文: a nonaqueous secondary battery comprising the separator according to any one of claims 1 to 8 .  
正解文: 請求項１～８のいずれかに記載のセパレータを用いた非水系二次電池。  
生成文: 請求項１～８のいずれかに記載のセパレータを用いた非水系二次電池。  
BLEU: 0.887015102729059  

元の文: a pharmaceutical product comprising the anti-aging agent according to claim 8 .  
正解文: 請求項８に記載の抗老化剤を含む医薬品。  
生成文: 請求項８に記載の抗老化剤を含む医薬。  
BLEU: 0.8857000285382948  

### Case 2: 2 Bidirectional LSTM layers + Small data set

元の文: a laminate obtained by curing the prepreg according to claim 20 .  
正解文: 請求項２０に記載のプリプレグを硬化して得られる、積層板。  
生成文: 請求項２０に記載のプリプレグを含む、硬化物。  
BLEU: 0.8694417438899827  

元の文: an ink-jet printer comprising the ink-jet head according to claim 9 .  
正解文: 請求項９に記載のインクジェットヘッドを備えたインクジェットプリンタ。  
生成文: 請求項９に記載のインクジェットを用いたインクジェット装置。  
BLEU: 0.8566209113168688  

元の文: the electrolyte membrane according to claim 1  wherein the polymer ( 2 ) is an aromatic polyether polymer or a fluorine-containing polymer .  
正解文: 前記重合体（２）が芳香族ポリエーテル系重合体または含フッ素ポリマーである、請求項１に記載の電解質膜。  
生成文: 前記高分子系支持体が（ａ）系樹脂物または請求項１に記載の電解質膜。  
BLEU: 0.8498912392268879  

元の文: a fluorinated ether composition containing at least 95 mass % of the fluorinated ether compound as defined in any one of claims 1 to 6 .  
正解文: 請求項１～６のいずれか一項に記載の含フッ素エーテル化合物を９５質量％以上含む、含フッ素エーテル組成物。  
生成文: 請求項１～６のいずれか一項に記載の含フッ素エーテル化合物を９５質量％以上含む、含フッ素エーテル組成物。  
BLEU: 0.8408964152537145  

元の文: carbon fiber reinforced composite material produced by curing prepreg as described in either claim 9 or 10 .  
正解文: 請求項９または１０に記載のプリプレグを硬化させて得られる炭素繊維強化複合材料。  
生成文: 請求項９または１０に記載のプリプレグを硬化させてなる炭素繊維強化炭素材料。  
BLEU: 0.8408964152537145  

### Case 1: 1 LSTM layer + Small data set

元の文: the vinyl chloride resin composition of claim 8 used in powder molding .  
正解文: 粉体成形に用いられる、請求項８に記載の塩化ビニル樹脂組成物。  
生成文: 請求項８に記載の粉体成形用樹脂組成物。  
BLEU: 0.8954237688029468  

元の文: an ink-jet printer comprising the ink-jet head according to claim 9 .  
正解文: 請求項９に記載のインクジェットヘッドを備えたインクジェットプリンタ。  
生成文: 請求項９に記載のインクジェットを用いたインク。  
BLEU: 0.8739351325046805  

元の文: the copper alloy wire according to any one of claims 1 to 5  wherein the wire diameter or the wire thickness is 50 µm or less .  
正解文: 線径または線材の厚さが５０μｍ以下である請求項１～５のいずれか１項に記載の銅合金線材。  
生成文: 前記線厚が５０μｍ以下である、請求項１～５のいずれか１項に記載の銅合金線材。  
BLEU: 0.8676247188209203  

元の文: a molded article comprising a polyamide resin composition according to any one of claims 25 to 27 .
正解文: 請求項２５～２７のいずれか一項に記載のポリアミド樹脂組成物を含む、成形体。  
生成文: 請求項２５～２７のいずれか一項に記載のポリアミド樹脂組成物を含む、成形体。  
BLEU: 0.8650615454144222  

元の文: a laminate obtained by curing the prepreg according to claim 20 .  
正解文: 請求項２０に記載のプリプレグを硬化して得られる、積層板。  
生成文: 請求項２０に記載の硬化物を硬化してなる硬化物。  
BLEU: 0.8529987544592307  


```python

```
