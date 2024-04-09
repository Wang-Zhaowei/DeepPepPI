import load_data
from sklearn.model_selection import KFold
from keras import optimizers
import Model
import test_scores as score


def DeepPepPI(species):
    pept_emb_file = './PepPI dataset/'+species+'/peptide embedding features from ProtT5.npy'
    pept_emb_dict = load_data.extract_pept_feat(pept_emb_file)

    prot_seq_file = './PepPI dataset/'+species+'/protein sequences.fasta'
    prot_str_file = './PepPI dataset/'+species+'/protein secondary structures.fasta'
    order = 2 
    prot_mat_dict = load_data.extract_prot_feat(prot_seq_file, prot_str_file, order)

    file_path = './PepPI dataset/'+species+'/PepPIs cv.txt'
    posi_pairs = load_data.load_pairs(file_path)

    file_path = './PepPI dataset/'+species+'/non-PepPIs cv.txt'
    nega_pairs = load_data.load_pairs(file_path)

    pept_emb_feat, prot_mat_feat, label = load_data.load_dataset(pept_emb_dict, prot_mat_dict, posi_pairs, nega_pairs)
    print(pept_emb_feat.shape, prot_mat_feat.shape, label.shape)

    eval_metrics = []
    n_fold = 5
    Kfold = KFold(n_splits=n_fold, shuffle=True)
    for train_index,test_index in Kfold.split(label):
        pept_emb_train, pept_emb_test = pept_emb_feat[train_index], pept_emb_feat[test_index]
        prot_mat_train, prot_mat_test = prot_mat_feat[train_index], prot_mat_feat[test_index]
        y_train, y_test = label[train_index], label[test_index]
        print(pept_emb_train.shape, prot_mat_train.shape, y_train.shape)
        print(pept_emb_test.shape, prot_mat_test.shape, y_test.shape)

        model = Model.DeepPepPI_model(pept_emb_train.shape, prot_mat_train.shape)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x = [pept_emb_train, prot_mat_train], y = y_train, batch_size=128, epochs=100, verbose=1,shuffle=True)
        y_prob = model.predict([pept_emb_test, prot_mat_test])[:,-1]
        
        tp, fp, tn, fn, acc, prec, recall, MCC, f1_score, AUC, AUPR = score.calculate_performace(y_prob, y_test[:,-1])
        eval_metrics.append([tp, fp, tn, fn, acc, prec, recall, MCC, f1_score, AUC, AUPR])
        print('\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t', recall, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC)

    ave_tp, ave_fp, ave_tn, ave_fn, ave_acc, ave_prec, ave_recall, ave_MCC, ave_f1_score, ave_AUC, ave_AUPR = score.get_average_metrics(eval_metrics)
    print('\n Acc = \t'+ str(ave_acc)+'\n prec = \t'+ str(ave_prec)+ '\n recall = \t'+str(ave_recall)+ '\n MCC = \t'+str(ave_MCC)+'\n f1_score = \t'+str(ave_f1_score)+'\n AUC = \t'+ str(ave_AUC) + '\n AUPR =\t'+str(ave_AUPR)+'\n')
    fw = open('./Results/'+species+' DeepPePPI results cv.txt', 'a+')
    fw.write('tp\t'+str(ave_tp)+'\tfp\t'+str(ave_fp)+'\ttn\t'+str(ave_tn)+'\tfn\t'+str(ave_fn)+'\tAcc\t'+str(ave_acc)+'\tPrec\t'+str(ave_prec)+'\tRec\t'+str(ave_recall)+'\tMCC\t'+str(ave_MCC)+'\tF1\t'+str(ave_f1_score)+'\tAUC\t'+str(ave_AUC)+'\tAUPR\t'+str(ave_AUPR)+'\n')


if __name__ == '__main__':
    species = 'Arabidopsis thaliana' #Arabidopsis thaliana; Solanum lycopersicum
    DeepPepPI(species)