以下の論文の追試用リポジトリ．  
https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes  
  
    
1. ShrinkWrappingParametrization  
・論文中で提案されているShrink wrapping parametrizationと  
・Autoencoderを通したモデルの可視化を行なうC++/CLIプロジェクト  
      
2. autoencoderフォルダ  
・autoencoder_shrinkwraping.py : pytorchによるautoencoder実装  
・autoencoder_shrinkwraping_convert_pt.py: autoencoderと重みを，.pt形式で書き出す  
・AutoencoderCarDll : .ptファイルをlibtorchによりdllかするプロジェクト  
  
  
動作の様子--> autoencoder_rand.mp4
  
  
  
追試した手順 のメモ（自分用） 
1.データ準備  
　梅谷さんのページより（http://www.nobuyuki-umetani.com/publication/2017_siggatb_explore/2017_siggatb_ExploringGenerative3Dshapes_Objs.zip）
　車のobjモデルをダウンロード．  
　これは車の4角形メッシュモデル群で，これをShrink wraping parametrizationすることで特徴ベクトルに変換する  
  
2. データの変換  
  車のobjモデルをすべて展開して，以下のフォルダに配置．  
  ./ShrinkWrappingParametrization/ShrinkWrappingParametrization/models  
  ShrinkWrappingParametrizationプロジェクトをコンパイルして実行すると，  
  model内のすべてのobjをパラメータかして同じフォルダにテキストケー式で保存する．  
  
  ※この実行には，glewとEigenをダウンロードして3rdpartyフォルダに配置する必要あり  
  ※TCoreのコンストラクタよりファイル変換を呼び出す．現在この部分はif(false)してある  
  
  
3. Autoencoderの学習  
　autoencoder_shrinkwraping.pyと同じフォルダにmodelsというフォルダを作り，2で生成されたすべての .txtファイルを入れる
  autoencoder_shrinkwraping.pyを実行すると学習が始まり，学習途中の重みデータが
  ./autoencoder_sw  
  フォルダにたまっていく． 
  
  学習後conversion.pyを実行すると，重みデータを読み込み，ネットワークと重みを.ptファイルとしてセーブする
  
  ※中間層のユニット数やDropoutの有無など色々変えて実行してみた  
  ※現在のところユニット数を20くらい取ると結構良い結果が出た  
  ※autoencoder_shrinkwraping.pyとconversion.pyが参照するフォルダは適宜書き換える必要あり  
  ※実行にはpytorchが必要  
  
  
4. libtorchにより オートエンコーダをC++で動かすdllを作成  
　libtorchをダウンロード、解凍して  
　./autoencoder/AutoencoderCarDll/AutoencoderCarDll/3rdparty  
  に配置してコンパイル．  
  ※細かなプロジェクトファイルの設定方法などは，このプロジェクトのAutoencoderCarDll.cppに記載．  
  ※警告は沢山出るがとりあえず動く。。。  
  
  生成された以下のファイルをしかるべきところへ配置  
  AutoencoderCarDll.dll  --> ShrinkWrappingParametrizationプロジェクトのexeと同じ場所へ  
  AutoencoderCarDll.lib  --> ShrinkWrappingParametrizationプロジェクトのmydllへ  
  AutoencoderForCarDll.h --> ShrinkWrappingParametrizationプロジェクトのmydllへ  
    
  
5. Autoencoderの精度テストと，decoderを利用した車のランダム生成    
　2で利用したShrinkWrappingParametrizationプロジェクトを利用する．  
  
  ./ShrinkWrappingParametrization/ShrinkWrappingParametrization フォルダ内に，3で作成した学習済みのモデルデータ(pt)を配置．  
  full.pt : Encoder --> Decoderのネットワーク  
  encoder.pt : Encoder のみのネットワーク  
  decoder.pt : Decoder のみのネットワーク  
  
  ※ "a"キーを押すとランダム値をオートエンコーダに与え，そこから自動的に車モデルを生成する．  
  
