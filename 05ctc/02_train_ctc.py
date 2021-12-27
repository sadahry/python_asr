# -*- coding: utf-8 -*-

#
# CTCを学習します．
#

import tensorflow as tf

# 作成したDatasetクラスをインポート
from my_dataset import build_dataset
from my_model import build_model

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# json形式の入出力を行うモジュールをインポート
import json

# os, sys, shutilモジュールをインポート
import os
import sys
import shutil

os.chdir(os.path.dirname(__file__))

def ctc_simple_decode(int_vector, token_list):
    ''' 以下の手順で，フレーム単位のCTC出力をトークン列に変換する
        1. 同じ文字が連続して出現する場合は削除
        2. blank を削除
    int_vector: フレーム単位のCTC出力(整数値列)
    token_list: トークンリスト
    output:     トークン列
    '''
    # 出力文字列
    output = []
    # 一つ前フレームの文字番号
    prev_token = -1
    # フレーム毎の出力文字系列を前から順番にチェックしていく
    for n in int_vector:
        if n != prev_token:
            # 1. 前フレームと同じトークンではない
            if n != 0:
                # 2. かつ，blank(番号=0)ではない
                # --> token_listから対応する文字を抽出し，
                #     出力文字列に加える
                output.append(token_list[n])
            # 前フレームのトークンを更新
            prev_token = n
    return output


#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'phone'

    # 学習データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_train = '../01compute_features/fbank/train_small'
    # 開発データの特徴量(Feats.scp)が存在するディレクトリ
    feat_dir_dev = '../01compute_features/fbank/dev'

    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train) 
    exp_dir = './exp_' + os.path.basename(feat_dir_train) 

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, 'feats.scp')
    feat_scp_dev = os.path.join(feat_dir_dev, 'feats.scp')

    # 学習/開発データのラベルファイル
    label_train = os.path.join(exp_dir, 'data', unit,
                               'label_'+train_set_name)
    label_dev = os.path.join(exp_dir, 'data', unit,
                             'label_dev')
    
    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(feat_dir_train, 'mean_std.txt')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, unit+'_model_ctc')

    # ミニバッチに含める発話数
    batch_size = 10

    # 最大エポック数
    max_num_epoch = 60

    # 中間層のレイヤー数
    num_layers = 5

    # 層ごとのsub sampling設定
    # [1, 2, 2, 1, 1]の場合は，2層目と3層目において，
    # フレームを1/2に間引く(結果的にフレーム数が1/4になる)
    sub_sample = [1, 2, 2, 1, 1]

    # RNNのタイプ(LSTM or GRU)
    rnn_type = 'GRU'

    # 中間層の次元数
    hidden_dim = 320

    # Projection層の次元数
    projection_dim = 320

    # bidirectional を用いるか(Trueなら用いる)
    bidirectional = True

    # 初期学習率
    initial_learning_rate = 1.0

    # Clipping Gradientの閾値
    clip_grad_threshold = 5.0

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

    # 学習過程で，認識エラー率を計算するか否か
    # 認識エラー率の計算は時間がかかるので注意
    # (ここではvalidationフェーズのみTrue(計算する)にしている)
    evaluate_error = {'train': False, 'validation': True}

    #
    # 設定ここまで
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'rnn_type': rnn_type,
              'num_layers': num_layers,
              'sub_sample': sub_sample, 
              'hidden_dim': hidden_dim,
              'projection_dim': projection_dim,
              'bidirectional': bidirectional,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'clip_grad_threshold': clip_grad_threshold,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold
             }

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

    # トークンリストをdictionary型で読み込む
    token_list = {}
    with open(token_list_path, mode='r') as f:
        # 1行ずつ読み込む
        for line in f: 
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            parts = line.split()
            # 0番目の要素がトークン，1番目の要素がID
            token_list[int(parts[1])] = parts[0]

    # トークン数
    # kerasのCTCでは最後のindexをblankとするため、token_listには含めずトークン合計数を +1 する
    # the +1 stands for a blank character needed in CTC
    # https://github.com/kutvonenaki/simple_ocr/blob/bb35e437d6b2447d084500c032461bf13c95cc3f/ocr_source/models.py#L62
    num_tokens = len(token_list) + 1

    pad_index = 0.0

    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = build_model(
        feat_dim,
        pad_index,
        num_layers,
        hidden_dim,
        bidirectional,
        projection_dim,
        num_tokens,
        rnn_type,
        initial_learning_rate,
    )

    model.summary()

    # 訓練/開発データのデータセットを作成する
    train_dataset = build_dataset(feat_scp_train,
                                    label_train,
                                    feat_mean,
                                    feat_std,
                                    batch_size,
                                    num_tokens,
                                    pad_index)

    # 開発データのデータセットを作成する
    dev_dataset = build_dataset(feat_scp_dev,
                                  label_dev,
                                  feat_mean,
                                  feat_std,
                                  batch_size,
                                  num_tokens,
                                  pad_index)

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

    # with tf.device("/cpu:0"): # faster than gpu when setting loss func in model.compile()
    history = model.fit(
        train_dataset,
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


