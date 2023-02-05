import pandas as pd

def main():

    df = pd.read_csv('metadata_compiled.csv')

    df2 = pd.DataFrame()

    df2['path'] = 'coughvid_data/' + df.loc[df['cough_detected'] > 0.5]['uuid'] + '.wav'

    df2['is_cough'] = 1

    df3 = pd.DataFrame()

    df3['path'] = 'coughvid_data/' + df.loc[df['cough_detected'] < 0.5]['uuid'] + '.wav'

    df3['is_cough'] = 0

    df = pd.concat([df2, df3], axis=0)

    df.to_csv('ML_labels.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    main()
