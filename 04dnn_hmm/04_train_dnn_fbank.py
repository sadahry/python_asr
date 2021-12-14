# -*- coding: utf-8 -*-

#
# DNNを学習します．
#

from six import b
import tensorflow as tf

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# json形式の入出力を行うモジュールをインポート
import json

# os, sys, shutilモジュールをインポート
import os
import sys
import shutil

os.chdir(os.path.dirname(__file__))

#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # 訓練データの特徴量リスト
    train_feat_scp = \
        '../01compute_features/fbank/train_small/feats.scp'
    # 訓練データのラベル(アライメント)ファイル
    train_label_file = \
        './exp/data/train_small/alignment'
    
    # 訓練データから計算された
    # 特徴量の平均/標準偏差ファイル
    mean_std_file = \
        '../01compute_features/fbank/train_small/mean_std.txt'

    # 開発データの特徴量リスト
    dev_feat_scp = \
        '../01compute_features/fbank/dev/feats.scp'
    # 開発データのラベル(アライメント)ファイル
    dev_label_file = \
        './exp/data/dev/alignment'

    # HMMファイル
    # HMMファイルは音素数と状態数の
    # 情報を得るためだけに使う
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 学習結果を出力するディレクトリ
    output_dir = os.path.join('exp', 'model_dnn_fbank')

    # ミニバッチに含める発話数
    batch_size = 5

    # 最大エポック数
    max_num_epoch = 60

    # 中間層のレイヤー数
    num_layers = 4

    # 中間層の次元数
    hidden_dim = 1024

    # splice: 前後 n フレームの特徴量を結合する
    # 次元数は(splice*2+1)倍になる
    splice = 5

    # 初期学習率
    initial_learning_rate = 0.008

    # 学習率の減衰やEarly stoppingの
    # 判定を開始するエポック数
    # (= 最低限このエポックまではどれだけ
    # validation結果が悪くても学習を続ける)
    lr_decay_start_epoch = 7

    # 学習率を減衰する割合
    # (減衰後学習率 <- 現在の学習率*lr_decay_factor)
    # 1.0以上なら，減衰させない
    lr_decay_factor = 0.5

    # Early stoppingの閾値
    # 最低損失値を更新しない場合が
    # 何エポック続けば学習を打ち切るか
    early_stop_threshold = 3

    #
    # 設定ここまで
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'num_layers': num_layers, 
              'hidden_dim': hidden_dim,
              'splice': splice,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold}

    # 設定をJSON形式で保存する
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w') as f:
        json.dump(config, f, indent=4)

    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy arrayに変換
        feat_mean = np.array(feat_mean, 
                             dtype=np.float32)
        feat_std = np.array(feat_std, 
                            dtype=np.float32)
    # 平均/標準偏差ファイルをコピーする
    shutil.copyfile(mean_std_file,
                    os.path.join(output_dir, 'mean_std.txt'))

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)

    # DNNの出力層の次元数を得るために，
    # HMMの音素数と状態数を得る
    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()
    # HMMを読み込む
    hmm.load_hmm(hmm_file)
    # DNNの出力層の次元数は音素数x状態数
    dim_out = hmm.num_phones * hmm.num_states
    # バッチデータ作成の際にラベルを埋める値
    # はdim_out以上の値にする
    pad_index = dim_out
    
    # ニューラルネットワークモデルを作成する
    # 入力特徴量の次元数は
    # feat_dim * (2*splice+1)
    dim_in = feat_dim * (2*splice+1)

    '''LeCunのパラメータ初期化方法の実行
    各重み(バイアス成分除く)を，平均0，標準偏差 1/sqrt(dim) の
    正規分布に基づく乱数で初期化(dim は入力次元数)
    model: Pytorchで定義したモデル
    '''
    initializer = tf.keras.initializers.LecunNormal()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(input_shape=(dim_in,), mask_value=pad_index))
    for _ in range(num_layers + 1):
        model.add(tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(dim_out, kernel_initializer=initializer))

    # # sigmoidでのモデル計算
    # model = tf.keras.Sequential()
    # model.add(tf.keras.Input(shape=(dim_in,)))
    # model.add(tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid, kernel_initializer=initializer))
    # model.add(tf.keras.layers.Dense(dim_out, kernel_initializer=initializer))

    model.summary()

    model.compile(
        # オプティマイザを定義
        # ここでは momentum stochastic gradient descent
        # を使用
        optimizer=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.99),
        # クロスエントロピーを損失関数として用いる
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # 訓練データのデータセットを作成する
    # padding_indexはdim_out以上の値に設定する
    train_dataset = SequenceDataset(train_feat_scp,
                                    train_label_file,
                                    feat_mean,
                                    feat_std,
                                    pad_index,
                                    splice)
    # 開発データのデータセットを作成する
    dev_dataset = SequenceDataset(dev_feat_scp,
                                  dev_label_file,
                                  feat_mean,
                                  feat_std,
                                  pad_index,
                                  splice)

    # ログファイルの準備
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                                 mode='w')
    log_file.write('epoch\ttrain loss\t'\
                   'train err\tvalid loss\tvalid err\n')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=early_stop_threshold,
    )

    # 損失値が最低値を更新した場合は，
    # その時のモデルを保存する
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir,'best_model'),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        # probably below is correct
        # monitor='val_accuracy',
        # mode='max',
        save_best_only=True,
    )

    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_file.write(f"{epoch}\t{logs['loss']}\t{1.0-logs['accuracy']}" \
                           f"\t{logs['val_loss']}\t{1.0-logs['val_accuracy']}\n")

    logging_callback = LoggingCallback()

    history = model.fit(
        train_dataset,
        # batch_sizeのように各発話をまとめる機構は省略
        # 実装するならSequenceDatasetに組み込む必要あり
        # > Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        validation_data=dev_dataset,
        epochs=max_num_epoch,
        callbacks=[early_stopping, model_checkpoint_callback, logging_callback],
    )

    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')

    # 最終エポックのモデルを保存する
    model.save(os.path.join(output_dir,'final_model'))

    metrics = history.history

    # 最終エポックの情報
    print('Final epoch model -> %s/final_model' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model\n' \
                   % (output_dir))
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        loss = 'val_loss' if phase == 'validation' else 'loss'
        print('    %s loss: %f' \
              % (phase, metrics[loss][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, metrics[loss][-1]))
        # 最終エポックのエラー率を出力    
        acc = 'val_accuracy' if phase == 'validation' else 'accuracy'
        error_rate = (1.0 - metrics[acc][-1]) * 100
        print('    %s error rate: %f %%' \
              % (phase, error_rate))
        log_file.write('    %s error rate: %f %%\n' \
                       % (phase, error_rate))

    best_epoch, best_score = max(enumerate(metrics['val_accuracy']), key = lambda x:x[1])

    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        loss = 'val_loss' if phase == 'validation' else 'loss'
        print('    %s loss: %f' \
              % (phase, metrics[loss][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, metrics[loss][best_epoch]))
        # ベストエポックのエラー率を出力
        acc = 'val_accuracy' if phase == 'validation' else 'accuracy'
        error_rate = (1.0 - metrics[acc][best_epoch]) * 100
        print('    %s error rate: %f %%' \
              % (phase, error_rate))
        log_file.write('    %s error rate: %f %%\n' \
            % (phase, error_rate))

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        loss = 'val_loss' if phase == 'validation' else 'loss'
        plt.plot(metrics[loss],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/loss.png')

    # 認識誤り率の履歴グラフにして保存する
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        acc = 'val_accuracy' if phase == 'validation' else 'accuracy'
        error_rates = (1.0 - np.array(metrics[acc])) * 100
        plt.plot(error_rates,
                 label=phase+' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir+'/error.png')

    # ログファイルを閉じる
    log_file.close()
