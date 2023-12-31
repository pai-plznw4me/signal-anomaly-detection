from keras.models import Model
from keras.layers import Input, LSTM, Dense
import pandas as pd
import numpy as np
from copy import copy
from sklearn import preprocessing


def load_valid_norm_datasets_sorted_by_date(csv_path):
    """
       정렬된 날짜별로 정규화된 데이터셋을 로드하는 함수.

       Parameters:
       - csv_path (str): CSV 파일 경로

       Returns:
       - target_df (pd.DataFrame): 원본 데이터셋
       - norm_target_df (pd.DataFrame): 정규화된 데이터셋
       - scalers (list): 각 열에 대한 Min Max Scaler 리스트

       Example:
       >>> csv_path = "path/to/your/data.csv"
       >>> data, norm_data, data_scalers = load_valid_norm_datasets_sorted_by_date(csv_path)
       >>> print(data.head())
          cell_01  cell_02  cell_03  ...  chg_charging_now  drv_cyc
       0      ...      ...      ...  ...               ...      ...
       1      ...      ...      ...  ...               ...      ...
       2      ...      ...      ...  ...               ...      ...
       3      ...      ...      ...  ...               ...      ...
       4      ...      ...      ...  ...               ...      ...
       """
    # 데이터를 로드 합니다.
    df = pd.read_csv(csv_path)

    # 사용할 columns 을 나열합니다.
    cols = ['cell_01', 'cell_02', 'cell_03', 'cell_04', 'cell_05', 'cell_06', 'cell_07', 'cell_08', 'cell_09',
            'cell_10',
            'cell_11', 'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17', 'cell_18', 'cell_19',
            'cell_20',
            'cell_21', 'cell_22', 'cell_23', 'cell_24', 'cell_25', 'cell_26', 'cell_27', 'cell_28', 'cell_29',
            'cell_30',
            'cell_31', 'cell_32', 'cell_33', 'cell_34', 'cell_35', 'cell_36', 'cell_37', 'cell_38', 'cell_39',
            'cell_40',
            'cell_41', 'cell_42', 'cell_43', 'cell_44', 'cell_45', 'cell_46', 'cell_47', 'cell_48', 'cell_49',
            'cell_50',
            'cell_51', 'cell_52', 'cell_53', 'cell_54', 'cell_55', 'cell_56', 'cell_57', 'cell_58', 'cell_59',
            'cell_60',
            'cell_61', 'cell_62', 'cell_63', 'cell_64', 'cell_65', 'cell_66', 'cell_67', 'cell_68', 'cell_69',
            'cell_70',
            'cell_71', 'cell_72', 'cell_73', 'cell_74', 'cell_75', 'cell_76', 'cell_77', 'cell_78', 'cell_79',
            'cell_80',
            'cell_81', 'cell_82', 'cell_83', 'cell_84', 'cell_85', 'cell_86', 'cell_87', 'cell_88', 'cell_89',
            'cell_90',
            'msr_data.ibm', 'msr_data.r_isol', 'msr_data.vb_max', 'msr_data.vb_min', 'msr_tbmax_raw', 'msr_tbmin_raw',
            'SOC', 'CF_OBC_DCChargingStat', 'chg_charging_now', 'drv_cyc']
    date_col = ['dnt']

    # 사용할 column 만 추출합니다.
    target_df = df.loc[:, cols]
    target_df.head(1)

    # 날짜 정보 추출합니다.
    date_df = df.loc[:, date_col]
    sorted_date_df = date_df.sort_values(by='dnt', ascending=True)

    # 날짜 순서로 데이터를 정렬합니다.
    sorted_date_df = date_df.sort_values(by='dnt', ascending=True)
    # DataFrame이 시계열 순서대로(오름차순) 나열되어 있는지 검증합니다.
    index = sorted_date_df.index.values
    offset = np.roll(index, 1)
    offset[0] = -1
    assert np.all(index - offset == 1)

    # 각 열(column) 별 Min Max Scaler 을 생성하고 Min Max 가 적용된 데이터를 추출합니다.
    scalers = []
    norm_target_df = copy(target_df)
    for index in range(len(target_df.columns)):
        # Min Max scaler 을 생성합니다.
        target_col = target_df.iloc[:, index:index + 1]
        scaler = preprocessing.MinMaxScaler().fit(target_col)
        scalers.append(scaler)

        # 생성된 Min Max Scaler 을 활용해서 데이터을 정규화 합니다.
        norm_target_df.iloc[:, index:index + 1] = scaler.transform(target_col)
    return target_df, norm_target_df, scalers

