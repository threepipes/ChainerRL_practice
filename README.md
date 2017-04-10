# 概要
(最終更新: 2017/4/10)

chainerの強化学習ライブラリが公開されていたので動かしてみました．

chainerRL (https://github.com/pfnet/chainerrl)

動作確認が目的だったので，強化学習の内容自体は簡潔です．

## 内容

2次元平面上で車を時計回りに走らせる．

車は座標(x, y)と向き(dir: ラジアン)を持っており，前進と方向転換ができる．

(0, 0)を中心にして，車が時計回りに回るように強化学習を行った．

## 実行確認環境

* Windows 10
* CPUのみ
* Python 3.6.0

### requirements

* chainer
* chainerrl
* numpy
* gym (chainerrlのインストールと同時に入る)
* pygame (結果描画のため．該当箇所を消せば入れなくても動作)

## 実行

main_training.pyを実行してください．

## ソースコード

### main_training.py

chainerRLの強化学習実行コードが書かれている．

ほぼチュートリアル(https://github.com/pfnet/chainerrl/blob/master/examples/quickstart/quickstart.ipynb)に沿っている．

### model.py

今回強化学習を行う環境が書かれている．
* Cource
  * 車を走らせる2次元平面
* Car
  * 車を表す
  * 座標と向きを持ち，コマンドを与えることで操作できる
  
### canvas.py

実行結果を描画する．

車の移動履歴を受け取り，画面に表示する．

履歴はキーボードの左右で操作可能．
