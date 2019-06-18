import shutil
from train import *
from convert import MIDItoMP3
from music21 import stream


# 忽略警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_notes(model, network_input, note_name, notes_len):
    randindex = np.random.randint(0, len(network_input) - 1)

    notedic = dict((i,j) for i, j in enumerate(note_name))    # 把刚才的整数还原成音调

    pattern = list(network_input[randindex])
    predictions = []

    #随机生成500个音符
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(notes_len)  # 归一化

        prediction = model.predict(prediction_input, verbose = 0)  # verbose = 0 为不在标准输出流输出日志信息

        index = np.argmax(prediction)
        result = notedic[index]
        predictions = np.append(predictions, result)

        # 往后移动
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return predictions


# 生成mid音乐
def create_music():
    notes = get_notes()
    notes_len = len(set(notes))
    note_name = sorted(set(i for i in notes))
    sequence_length = 100  # 序列长度
    note_dict = dict((j, i) for i, j in enumerate(note_name))  # 设计一个字典，把音符转换成数字，方便训练
    network_input = []  # 创建输入序列
    network_output = []  # 创建输出序列

    for i in range(0, len(notes) - sequence_length):
        # 输入100个，输出1个
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_dict[k] for k in sequence_in])
        network_output.append(note_dict[sequence_out])

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normal_network_input = network_input / float(notes_len)  # 归一化

    # 自己训练模型
    # 寻找loss最小的weight文件，作为训练参数
    files = os.listdir()
    minloss = {}

    for i in files:
        if 'weights' in i:
            num = i[11:15]
            minloss[num] = i
    best_weights = minloss[min(minloss.keys())]
    print('最佳模型文件为:' + best_weights)
    model = get_model(normal_network_input, notes_len, best_weights)


    # 用已经训练好的模型
    # model = get_model(normal_network_input, notes_len, weights_file='weightsfilepath') # 填入模型文件名称

    predictions = generate_notes(model, network_input, note_name, notes_len)

    offset = 0
    output_notes = []

    # 生成 Note（音符）或 Chord（和弦）对象
    for data in predictions:
        # data is a chord
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes = np.append(notes, new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes = np.append(output_notes, new_chord)
        # data is a note
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes = np.append(output_notes, new_note)

        # 音符位置向后顺序移一位
        offset += 1

    # 创建音乐流（Stream）
    midi_stream = stream.Stream(output_notes)
    # 写入 MIDI 文件
    midi_stream.write('midi', fp='output.mid')


if __name__ == '__main__':
    create_music()
    for i in range(1,100):
        if os.path.exists('output%d.mid' % i):
            i += 1
        else:
            shutil.move("output.mid", "output%d.mid" % i)
            MIDItoMP3("output%d.mid" % i)
            break
