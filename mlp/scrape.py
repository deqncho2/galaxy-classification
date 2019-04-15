import pprint

def main():
    mins = dict()
    for learning_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for batch_size in [32, 64, 128, 192, 256]:
            for n in [2, 3, 4, 5]:
                mins[(n, batch_size, learning_rate)] = (0., 0)
    n = 0
    l = 0.
    b = 0
    epoch = 0
    #filename = 'binky.whistles'
    #filename = 'log.dog'
    filename = 'vanialla.test'
    for line in open(filename, 'r').readlines():
        line = line.strip()
        if line.startswith("N_HIDDEN_LAYERS"):
            n = int(line.split()[2])
            l = float(line.split()[5])
            b = int(line.split()[8])
            epoch = 0
        if line.startswith("error(train)") and n != 0.:
            epoch += 1
            v = float(line.split()[3][len("acc(valid)="):])
            #print(line.split()[3][len("acc(valid)="):])
            mins[(n, b, l)] = max(mins[(n, b, l)], (v, epoch), key=lambda x : x[0])

    outl = []
    for k, v in mins.iteritems():
        outl.append((k, v))
    outl = sorted(outl, key=lambda x : x[1][0])
    #outl = list(filter(lambda  x : x[1][1] >= 98, outl))
    #outl = list(map(lambda x : x[0][0], outl))
    #print(sum(map(lambda x : x[1][1], outl)) / float(len(outl)))
    #print(outl[-5:])
    true_mins = {
        2 : (0., 0, 0, 0),
        3 : (0., 0, 0, 0),
        4 : (0., 0, 0, 0),
        5 : (0., 0, 0, 0)
    }
    for batch_size in [32, 64, 128, 192, 256]:
        for n_hidden_layers in [2, 3, 4, 5]:
            for learning_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                    true_mins[n_hidden_layers] = \
                    max(true_mins[n_hidden_layers], (mins[(n_hidden_layers, batch_size, learning_rate)][0],
                                                    mins[(n_hidden_layers, batch_size, learning_rate)][1],
                                                    batch_size, learning_rate), key=lambda x : x[0])
                #mins[(batch_size, n_hidden_layers, learning_rate)] = (100., 0)
    dels = []
    for k, v in mins.iteritems():
        if v == (100, 0.):
            dels.append(k)
    for dd in dels:
        del mins[dd]
    print(true_mins)
    #pprint.PrettyPrinter(indent=4).pprint(mins)

if __name__ == "__main__":
    main()