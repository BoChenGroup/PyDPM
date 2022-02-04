import numpy as np
import copy

class NMI(object):
    def __init__(self, A, B):
        '''
        Inputs:
            A: [int], ground_truth, shape:(n_sample,)
            B: [int], pred_label, shape:(n_sample,)

        Outputs:
            NMI: [float], Normalized Mutual information of A and B

        '''
        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)

        self._get()

        print(f'The NMI is: {self._NMI:.4f}')

    def _get(self):
        n_gnd = self.A.shape[0]
        n_label = self.B.shape[0]
        # assert n_gnd == n_label

        LabelA = np.unique(self.A)
        nClassA = len(LabelA)
        LabelB = np.unique(self.B)
        nClassB = len(LabelB)

        if nClassB < nClassA:
            self.A = np.concatenate((self.A, LabelA))
            self.B = np.concatenate((self.B, LabelA))
        else:
            self.A = np.concatenate((self.A, LabelB))
            self.B = np.concatenate((self.B, LabelB))

        G = np.zeros([nClassA, nClassA])
        for i in range(nClassA):
            for j in range(nClassA):
                G[i, j] = np.sum((self.A == LabelA[i]) * (self.B == LabelA[j]))

        sum_G = np.sum(G)
        PA = np.sum(G, axis=1)
        PA = PA/sum_G
        PB = np.sum(G, axis=0)
        PB = PB/sum_G
        PAB = G/sum_G

        if np.sum((PA == 0)) > 0 or np.sum((PB == 0)):
            print('error ! Smooth fail !')
            self._NMI = np.nan
        else:
            HA = np.sum(-PA * np.log2(PA))
            HB = np.sum(-PB * np.log2(PB))
            PPP = PAB / np.tile(PB, (nClassA, 1)) / np.tile(PA.reshape(-1, 1), (1, nClassA))
            PPP[np.where(abs(PPP) < 1E-12)] = 1  # avoid 'log 0'
            MI = np.sum(PAB * np.log2(PPP))
            NMI = MI / np.max((HA, HB))
            # optional
            # NMI = 2.0 * MI / (HA + HB)
            # NMI = MI / np.sqrt(HA * HB)
            self._NMI = NMI

