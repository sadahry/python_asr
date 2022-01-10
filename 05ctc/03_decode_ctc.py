# -*- coding: utf-8 -*-

#
# CTCによるデコーディングを行います
#

import tensorflow as tf

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# json形式の入出力を行うモジュールをインポート
import json

# os, sysモジュールをインポート
import os
import sys

os.chdir(os.path.dirname(__file__))

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

    # 実験ディレクトリ
    exp_dir = './exp_train_small'

    # 評価データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_test = '../01compute_features/fbank/test'

    # 評価データの特徴量リストファイル
    feat_scp_test = os.path.join(feat_dir_test, 'feats.scp')

    # 評価データのラベルファイル
    label_test = os.path.join(exp_dir, 'data', unit, 'label_test')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')

    # 学習済みモデルが格納されているディレクトリ
    model_dir = os.path.join(exp_dir, unit+'_model_ctc')

    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(model_dir, 'mean_std.txt')

    # 学習済みのモデルファイル
    rnn_dir = os.path.join(model_dir, 'best_model')

    # デコード結果を出力するディレクトリ
    output_dir = os.path.join(model_dir, 'decode_test')

    # デコード結果および正解文の出力ファイル
    hypothesis_file = os.path.join(output_dir, 'hypothesis.txt')
    reference_file = os.path.join(output_dir, 'reference.txt')

    # 学習時に出力した設定ファイル
    config_file = os.path.join(model_dir, 'config.json')

    # ミニバッチに含める発話数
    batch_size = 8
    
    #
    # 設定ここまで
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

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

    # 学習済みのDNNファイルから
    # モデルを読み込む
    model = tf.keras.models.load_model(rnn_dir)

    # 予測モデルではCTCLayerを省いたモデルを利用
    model = tf.keras.models.Model(
        model.get_layer(name="feat").input, model.get_layer(name="softmax").output
    )

    model.summary()

    # 以下から踏襲
    # https://github.com/keras-team/keras-io/blob/2f3d2456d894f6669850a4336eed75b717ed0e7e/examples/vision/captcha_ocr.py#L301-L312

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # 文中のblankはdecode時に省いてくれるため、
        # たとえば本来 [<blank>,1,1,<blank>,<blank>,2,<blank>,1,<blank>,<blank>,] となる結果は [1,2,1,<blank>,<blank>,] となる
        # Use greedy search. For complex tasks, you can use beam search
        result = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :np.int64(max(input_len))
        ]
        return result

    # デコード結果および正解ラベルをファイルに書き込みながら
    # 以下の処理を行う
    with open(hypothesis_file, mode='w') as hyp_file, \
         open(reference_file, mode='w') as ref_file:
        # 評価データのfeat_scp_testから1ファイル
        # ずつ取り出して処理する．

        # 特徴量リスト，ラベルを1行ずつ
        # 読み込みながら情報を取得する
        with open(feat_scp_test, mode='r') as file_f, \
             open(label_test, mode='r') as file_l:
            for (line_feats, line_label) in zip(file_f, file_l):
                # 各行をスペースで区切り，
                # リスト型の変数にする
                parts_feats = line_feats.split()
                parts_label = line_label.split()

                utt_id = parts_feats[0]
                utt_id_label = parts_label[0]
                feat_path = parts_feats[1]
                label = np.int64(parts_label[1:])

                # 発話ID(partsの0番目の要素)が特徴量と
                # ラベルで一致していなければエラー
                if utt_id != utt_id_label:
                    sys.stderr.write('IDs of feat and '\
                        'label do not match.\n')
                    exit(1)

                feat = np.fromfile(feat_path, 
                                   dtype=np.float32)
                # フレーム数 x 次元数の配列に変形
                feat = feat.reshape(-1, feat_dim)

                # 平均と標準偏差を使って正規化(標準化)を行う
                feat = (feat - feat_mean) / feat_std

                # モデルの出力を計算(フォワード処理)
                # 1発話のみのバッチとして扱う
                preds = model.predict(np.array([feat]))
                pred_idxs = decode_batch_predictions(preds)
                pred_idxs = pred_idxs[0].numpy()
                # blankを削除
                # decode時にblankは -1 となる
                # Important: blank labels are returned as -1.
                # https://docs.w3cub.com/tensorflow~python/tf/keras/backend/ctc_decode
                pred_idxs = pred_idxs[pred_idxs != -1]

                pred_tokens = [token_list[pred_idx] for pred_idx in pred_idxs]
                label_tokens = [token_list[l] for l in label]

                # 結果を書き込む
                hyp_file.write('%s %s\n' \
                    % (utt_id, ' '.join(pred_tokens)))
                ref_file.write('%s %s\n' \
                    % (utt_id, ' '.join(label_tokens)))
