from matplotlib import pyplot as plt

detectorName='LSTM'

pwd = './results/'+detectorName+'/realKnownCause/'+detectorName+'_nyc_taxi.csv'
#pwd = './results/'+detectorName+'/realKnownCause/'+detectorName+'_machine_temperature_system_failure.csv'
#pwd = './results/'+detectorName+'/realKnownCause/'+detectorName+'_ambient_temperature_system_failure.csv'
#pwd = './results/'+detectorName+'/artificialWithAnomaly/'+detectorName+'_art_daily_jumpsup.csv'
#pwd = './results/'+detectorName+'/artificialWithAnomaly/'+detectorName+'_art_daily_flatmiddle.csv'
#pwd = './results/'+detectorName+'/artificialNoAnomaly/'+detectorName+'_art_daily_perfect_square_wave.csv'
#pwd = './results/'+detectorName+'/realTraffic/'+detectorName+'_speed_7578.csv'

#pwd = './results/'+detectorName+'/artificialWithAnomaly/'+detectorName+'_art_load_balancer_spikes.csv'

anotherPredValue=True

if anotherPredValue:
    times = []
    values = []
    anomaly_scores = []
    predValues1 = []
    predValues2 = []
    labels = []
    with open(pwd, 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line_splited = line.strip().split(',')
                time = line_splited[0]
                times.append(line_splited[0])
                values.append(float(line_splited[1]))
                anomaly_scores.append(float(line_splited[2]))
                predValues1.append(float(line_splited[3]))
                predValues2.append(float(line_splited[4]))

                labels.append(int(line_splited[5]))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(values, '.r')
    plt.plot(predValues1, '.b')
    plt.plot(predValues2, '.g')
    plt.subplot(2, 1, 2)
    plt.plot(labels, '.r')
    plt.plot(anomaly_scores, 'b')
    plt.show()
else:

    times = []
    values = []
    anomaly_scores = []
    predValues1 = []
    labels = []
    with open(pwd,'r') as file:
        for i, line in enumerate(file):
            if i>0:
                line_splited = line.strip().split(',')
                time=line_splited[0]
                times.append(line_splited[0])
                values.append(float(line_splited[1]))
                anomaly_scores.append(float(line_splited[2]))
                predValues1.append(float(line_splited[3]))
                labels.append(int(line_splited[4]))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(values,'.r')
    plt.plot(predValues1, '.b')
    plt.subplot(2,1,2)
    plt.plot(labels,'.r')
    plt.plot(anomaly_scores,'b')
    plt.show()