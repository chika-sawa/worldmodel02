# 「世界モデルと知能」最終課題で2グループ が格納したデータの説明及び構築手順、学習、評価の実行方法の説明

## 格納したデータについて
* eval評価結果valid_seen1.txt<br/>
818個のTaskについての評価の実行結果 seen環境​

* eval評価結果valid_seen2.txt<br/>
同上 2回目の実行結果​

* eval評価結果valid_unseen1.txt<br/>
818個のTaskについての評価の実行結果 unseen環境​

* eval評価結果valid_unseen2.txt<br/>
同上 2回目の実行結果​

* eval評価結果fast_epoch版.txt<br/>
debug用の10個のTask評価を行うfast_epoch実行結果​

* TensorBordグラフ.png<br/>
学習状況を示すTensorBordのグラフ画像キャプチャ​

* python関連のインストール済みパッケージ一覧.txt<br/>
パッケージバージョンの参考用

## コードについて
コードは次のGitHubのリポジトリのコードと同一のものを使用して、再現実装を行いました。<br/>
  https://github.com/askforalfred/alfred<br/>
  オリジナルのコードでバグ修正等が実施される可能性もあることから、このコードの格納は行いません。<br/>
  コードについては、上記のリポジトリを参照して下さい。

## 環境構築
環境構築について、以下、Cloudインスタンス環境での構築方法について説明します。<br/>
### Cloudインスタンスを作成する​
- インスタンス：Alibaba Cloud ECS(Elastic Compute Service)​
- GPU： NVIDIA T4 (NVIDIA Tesla T4)​
- CPU：16 vCores ​
- ストレージ：100 GB　※後述するデータセットの容量が大きく90-100G必要​
- Image：Ubuntu 16.04​
- GPUドライバ：CUDA ver10.0※インスタンス作成時に「GPU ドライバーの自動インストール」にチェックをして、 CUDA ver10.0を選択​
- OPENするPORT：追加で80,443のPORTを追加する。デフォルトではPORTが空いていないので追加。tensorbordを使用する場合は、PORT 6006も追加する。

### pythonコードのcloneとライブラリのインストール
alfredリポジトリをクローン​<br/>
```git clone https://github.com/askforalfred/alfred.git alfred​```<br/>
```export ALFRED_ROOT=$(pwd)/alfred​```<br/>

​必要なライブラリをインストール​<br/>
```cd $ALFRED_ROOT​```<br/>
```pip install --upgrade pip​```<br/>
```pip install -r requirements.txt​```<br/>

### データセットのダウンロード及び解凍
dataディレクトリに移動​<br/>
```cd $ALFRED_ROOT/data​```<br/>
ダウンロード及び解凍​<br/>
```sh download_data.sh json_feat```<br/>
※Alfred Githubのreadmeには、最大17Gと記載がありますが、ダウンロードした7zip形式の圧縮ファイルが17Gで、解答すると35Gあり、合計で52Gあります

### コードの修正
2022年4月現在、評価(eval)を実行時に、‘use_templated_goals’のエラーが発生するため、<br/>
その対応のため、alfred/data/preprocess.py“, line 67　をコメントアウトして以下を記載。​<br/>
```use_templated_goals = False```<br/>
※use_templated_goalsはオプションで指定しない限りFalseが設定されるはずなのですが、正しく設定できないバグのように見うけられます。​

### cuDNNエラーへの対応​
cuDNNがインストールされていない環境、または、 cuDNNがインストールされているけどバージョンが環境とマッチしていない場合は、 ```RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED```が発生します。​<br/>

対応としては、 以下のいずれかを実施します。​<br/>
※ NVIDIA cuDNNはGPUの処理を高速化するライブラリですので、可能であれば、cuDNNをインストールしたほうが高速に処理できます。

- 対応①<br>
  環境にマッチしたcuDNNをインストールする​<br/>
  Nvidiaのサイトでcudaのバージョンに対応したバージョンを調べcuDNNをインストールします。 NvidiaのDeveloperサイトからNvidiaアカウントでログインしてパッケージをインストールする必要があります。 Nvidiaアカウントはメールアドレスで作成できます。​

- 対応②<br>
  cuDNNを無効にする​<br/>
  models/eval/eval.pyの最初のほうに、以下のコードを追加する。​<br/>
  ```torch.backends.cudnn.enabled = False​```<br/>

## 学習 trainの実行
モデルの学習を行うには、 train_seq2seq.pyを実行します。<br/>
```cd $ALFRED_ROOT​```<br/>
以下のコマンドを実行​<br/>
```python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --preprocess​```<br/>
※初回は--preprocessオプションをつける(初回はtrain,evalともに必要)

## 学習済みモデルについて
学習済みモデルをダウンロードするには、以下を実行します。​<br>
​ダウンロード​<br/>
```wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/seq2seq_pm_chkpt.zip​```<br/>
解凍​<br/>
```unzip seq2seq_pm_chkpt.zip​```<br/>
alfredディレクトリに「model:seq2seq_im_mask,name:base30_pm010_sg010_01」ディレクトリが作成されれば成功です。

## 評価 evalの実行
### ディスプレイの設定
Cloudインスタンス環境では、そのまま実行するとディスプレイが無いのでエラーが発生します。次の対応を行い、ディスプレイが無い場合の設定を行います。​<br>
Xorgをインストールします。​<br>
```sudo apt-get update```<br/>
```sudo apt-get install -y xserver-xorg mesa-utils```<br/>​
```sudo nvidia-xconfig -a --use-display-device = None --virtual = 1280x1024​```<br/>

以下のコマンドでBusID情報を取得し、BusIDを書き留めます。例) PCI:0:7:0​<br/>
```nvidia-xconfig --query-gpu-info​```<br/>
以下のコマンドを実行してXorgを構成します。 PCI:0:7:0は、自分の環境に合わせます。<br/>​
```sudo Xorg -noreset -sharevts -novtswitch -isolateDevice "PCI:0:7:0" -config xorg.conf :0 vt1​```<br/>
※複数のPCIがあるときは、対応するXorgを複数構成します。<br>
tmuxがインストールされていない場合はインストールします。<br>

### X serverの起動
続いて、 X serverの起動を行います。​<br>
tmux sessionをスタートします。​<br>
```tmux new -s startx ​```<br>
tmux shell (X server)が、DISPLAY 0で起動します。以下のコードを実行します。​<br>
```sudo python $ALFRED_ROOT/scripts/startx.py​```<br>
tmux shellを Ctrl+b、続いて d を入力して、終了します。​<br>
X serverのDISPLAY変数に0を設定します。​<br>
```export DISPLAY=:0​```<br>
```cd $ALFRED_ROOT​```<br>

以下のコマンドを実行して、THORの設定をチェックします。​<br>
```python scripts/check_thor.py​```<br>
以下のとおり表示されたら正しく設定されています。​<br>

```###############​```<br>
```## (300, 300, 3)​```<br>
```## Everything works!!!​```<br>

### eval_seq2seq.pyの実行
モデルの評価を行うには、eval_seq2seq.pyを実行します。<br>
※初回は--preprocessオプションをつける(初回はtrain,evalともに必要)<br>
- seen環境​<br>
```python models/eval/eval_seq2seq.py --model_path model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_seen.pth --eval_split valid_seen --data data/json_feat_2.1.0 --model models.model.seq2seq_im_mask --gpu --num_threads 1 --preprocess​```<br>

- unseen環境​<br>
```python models/eval/eval_seq2seq.py --model_path model:seq2seq_im_mask,name:base30_pm010_sg010_01/best_unseen.pth --eval_split valid_unseen --data data/json_feat_2.1.0 --model models.model.seq2seq_im_mask --gpu --num_threads 1 --preprocess​```<br>
