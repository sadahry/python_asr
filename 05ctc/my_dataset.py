# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

from numpy.lib.arraypad import pad
import tensorflow as tf

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# sysモジュールをインポート
import sys
import math

def build_dataset(feat_scp, 
                label_scp, 
                feat_mean, 
                feat_std,
                batch_size,
                num_tokens,
                pad_index=0,
                splice=0,
                ):
    ''' ミニバッチデータを作成するクラス
        torch.utils.data.Datasetクラスを継承し，
        以下の関数を定義する
        __len__: 総サンプル数を出力する関数
        __getitem__: 1サンプルのデータを出力する関数
    feat_scp:  特徴量リストファイル
    label_scp: ラベルファイル
    feat_mean: 特徴量の平均値ベクトル
    feat_std:  特徴量の次元毎の標準偏差を並べたベクトル 
    pad_index: バッチ化の際にフレーム数を合わせる
               ためにpaddingする整数値
    splice:    前後(splice)フレームを特徴量を結合する
               splice=1とすると，前後1フレーム分結合
               するので次元数は3倍になる．
               splice=0の場合は何もしない
    '''

    # 発話の数
    num_utts = 0
    # 各発話のID
    id_list = []
    # 各発話の特徴量ファイルへのパスを記したリスト
    feat_list = []
    # 各発話の特徴量フレーム数を記したリスト
    feat_len_list = []
    # 特徴量の平均値ベクトル
    feat_mean = feat_mean
    # 特徴量の標準偏差ベクトル
    feat_std = feat_std
    # 標準偏差のフロアリング
    # (0で割ることが発生しないようにするため)
    feat_std[feat_std<1E-10] = 1E-10
    # 特徴量の次元数
    feat_dim = \
        np.size(feat_mean)
    # 各発話のラベル
    label_list = []
    # 各発話のラベルの長さを記したリスト
    label_len_list = []
    # フレーム数の最大値
    max_feat_len = 0
    # ラベル長の最大値
    max_label_len = 0
    # 1バッチに含める発話数
    batch_size = batch_size
    # トークン数(blankを含む)
    num_tokens = num_tokens
    # フレーム埋めに用いる整数値
    pad_index = pad_index
    # splice:前後nフレームの特徴量を結合
    splice = splice

    # 特徴量リスト，ラベルを1行ずつ
    # 読み込みながら情報を取得する
    with open(feat_scp, mode='r') as file_f, \
            open(label_scp, mode='r') as file_l:
        for (line_feats, line_label) in zip(file_f, file_l):
            # 各行をスペースで区切り，
            # リスト型の変数にする
            parts_feats = line_feats.split()
            parts_label = line_label.split()

            # 発話ID(partsの0番目の要素)が特徴量と
            # ラベルで一致していなければエラー
            if parts_feats[0] != parts_label[0]:
                sys.stderr.write('IDs of feat and '\
                    'label do not match.\n')
                exit(1)

            # 発話IDをリストに追加
            id_list.append(parts_feats[0])
            # 特徴量ファイルのパスをリストに追加
            feat_list.append(parts_feats[1])
            # フレーム数をリストに追加
            feat_len = np.int32(parts_feats[2])
            feat_len_list.append(feat_len)

            # ラベル(番号で記載)をint型の
            # numpy arrayに変換
            label = np.int32(parts_label[1:])
            # ラベルリストに追加
            label_list.append(label)
            # ラベルの長さを追加
            label_len_list.append(np.int32(len(label)))

            # 発話数をカウント
            num_utts += 1

    # フレーム数の最大値を得る
    max_feat_len = \
        np.max(feat_len_list)
    # ラベル長の最大値を得る
    max_label_len = \
        np.max(label_len_list)

    # ラベルデータの長さを最大フレーム長に
    # 合わせるため，pad_indexの値で埋める
    for n in range(num_utts):
        # 埋めるフレームの数
        # = 最大フレーム数 - 自分のフレーム数
        pad_len = max_label_len \
                - label_len_list[n]
        label_list[n] = \
            np.pad(label_list[n], 
                    [0, pad_len], 
                    mode='constant', 
                    constant_values=pad_index)

    dataset = tf.data.Dataset.from_tensor_slices(
        (feat_list, label_list, feat_len_list, label_len_list))

    @tf.function
    def __getitem__(feat_path, label, feat_len, label_len):
        ''' サンプルデータを返す関数
        本実装では発話単位でバッチを作成する．
        '''
        feat = getfeat(feat_path, feat_len)

        # 特徴量，ラベルとそれぞれの長さのバッチを返す
        # paddingや変換によって元の長さが消失してしまうためinputに含める
        return {
            "feat": feat,
            "label": label,
            "feat_len": feat_len,
            "label_len": label_len,
        }

    def getfeat(feat_path, feat_len):
        ''' 特徴量の取得処理を分離
        '''
        # 特徴量データを特徴量ファイルから読み込む
        feat = tf.io.read_file(feat_path)
        feat = tf.io.decode_raw(feat, tf.float32)

        # フレーム数 x 次元数の配列に変形
        feat = tf.reshape(feat, (-1, feat_dim))

        # 平均と標準偏差を使って正規化(標準化)を行う
        feat = (feat - feat_mean) / feat_std

        # 特徴量データのフレーム数を最大フレーム数に
        # 合わせるため，pad_indexで埋める
        pad_len = max_feat_len - feat_len
        feat = tf.concat([feat, tf.fill((pad_len, feat_dim), pad_index)], 0)
        return feat

    dataset = (
        dataset.map(
            __getitem__,num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset
