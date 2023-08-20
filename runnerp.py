#!/usr/bin/env python
"""
Reads a VCF file, estimates some simulation parameters, and simulates via msprime.
NOTE: This assumes that the user has previously QC'd and filtered the VCF.
"""

from imports import *
from helpers import *
from manager import *
from simulator import *
from sequenceBatchGenerator import *
from networks import *

def pr(s):
    print('*'*100)
    print('*'*100)
    print('*'*100)
    print('*'*20,end='\t'*8)
    print('*'*20)
    print('*'*20,end='\t'*8)
    print('*'*20)
    l=(50-len(s)//2)//8
    print('',end='\t'*l)
    print(s)
    print('*'*20)
    print('*'*20,end='\t'*8)
    print('*'*20)
    print('*'*20,end='\t'*8)
    print('*'*20)
    print('*'*100)
    print('*'*100)
    print('*'*100)






def simualte_rel(args):
    

    ## Set seed
    if args.seed:
        os.environ['PYTHONHASHSEED']=str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    

    ## Set number of cores
    if args.nCPU_sim:
        nProc = args.nCPU_sim
    else:
        nProc = mp.cpu_count()
    
    
    ## Ensure all required arguments are provided
    if not args.vcf.endswith(".vcf"):
        print('Error: VCF file must end in extension ".vcf"')
        sys.exit(1)
    if not args.outDir:
        print("Warning: No project directory found, using current working directory.")
        projectDir = os.getcwd()
    else:
        projectDir = args.outDir
    if not args.mask:
        print("Warning: no accessibility mask found. All sites in the genome are assumed to be accessible.") 
    if args.dem:
        demHist = check_demHist(args.dem)
        if demHist == -9:
            print("Error: demographicHistory file must be raw output from either stairwayplot, SMC++, or MSMC")
            print("If using SMC++, file must be in *.csv format (use option -c in SMC++)")
            sys.exit(1)
        if not args.genTime:
            print("Error: assumed generation time must be supplied when simulating under stairwayplot, SMC++, or MSMC")
            sys.exit(1)
    else:
        print("Warning: no demographic history file found. All training data will be simulated under demographic equilibrium.")
        demHist = 0
    if not args.phased and args.phaseError != 0.0:
        print("Error: non-zero 'phaseError' cannot be used in conjunction with '--unphased'")
        sys.exit(1)
    if args.forceDiploid:
        print("Warning: all haploid/hemizygous samples will be treated as diploid samples with missing data!\n",
                "If you want to treat haploid/hemizygous samples and haploids without missing data, quit now, ensure no diploid samples are found in this VCF, and rerun without the option `--forceDiploid`.")
        time.sleep(10)
    else:
        time.sleep(5)
    
    
    ## Set up the directory structure to store the simulations data.
    trainDir = os.path.join(projectDir,"train")
    valiDir = os.path.join(projectDir,"vali")
    testDir = os.path.join(projectDir,"test")
    networkDir = os.path.join(projectDir,"networks")
    vcfDir = os.path.join(projectDir,"splitVCFs")


    ## Make directories if they do not exist
    for p in [projectDir,trainDir,valiDir,testDir,networkDir,vcfDir]:
        if not os.path.exists(p):
            os.makedirs(p)

    
    ## Read the genome file
    chromosomes = []
    with open(args.genome, "r") as fIN:
        for line in fIN:
            ar = line.split()
            if len(ar)!=3:
                print("Error: genome file must be formatted as a bed file (i.e.'chromosome     start     end')")
                sys.exit(1)
            chromosomes.append("{}:{}-{}".format(ar[0],ar[1],ar[2]))
   

    ## Pass params to the vcf manager    
    manager_params = {
            'vcf':args.vcf,
            'mask':args.mask,
            'winSizeMx':args.winSizeMx,
            'forceWinSize':args.forceWinSize,
            'forceDiploid':args.forceDiploid,
            'chromosomes':chromosomes,
            'vcfDir':vcfDir,
            'projectDir':projectDir,
            'networkDir':networkDir,
            'seed':args.seed
              }
    vcf_manager = Manager(**manager_params)
    
    
    ## Split the VCF file
    vcf_manager.splitVCF(nProc=nProc)
    

    ## Calculate nSites per window
    wins, nSamps, maxS, maxLen = vcf_manager.countSites(nProc=nProc)


    ## Prepare the accessibility mask
    if args.mask:
        mask_fraction, win_masks = vcf_manager.maskWins(wins=wins, maxLen=maxLen, nProc=nProc)
    else:
        mask_fraction, win_masks = 0.0, None
    
    
    ## Prepare the missing data mask
    md_mask, mask_files = None, []
    for FILE in glob.glob(os.path.join(vcfDir, "*_md_mask.hdf5")):
        mask_files.append(FILE)
        md_mask = []
    for FILE in mask_files:
        print("Reading HDF5 mask: {}...".format(FILE))
        with h5py.File(FILE, "r") as hf:
            md_mask.append(hf["mask"][:])
    if md_mask:
        md_mask = np.concatenate(md_mask)
    
    
    ## Define parameters for msprime simulation
    print("Simulating with window size = {} bp.".format(maxLen))
    a=0
    for i in range(nSamps-1):
        a+=1/(i+1)
    thetaW=maxS/a
    assumedMu = args.mu
    Ne=thetaW/(4.0 * assumedMu * ((1-mask_fraction) * maxLen))
    rhoHi=assumedMu*args.upRTR
    if demHist:
        MspD = convert_demHist(args.dem, nSamps, args.genTime, demHist, assumedMu)
        dg_params = {
                'priorLowsRho':0.0,
                'priorHighsRho':rhoHi,
                'priorLowsMu':assumedMu * 0.66,
                'priorHighsMu':assumedMu * 1.33,
                'ChromosomeLength':maxLen,
                'winMasks':win_masks,
                'mdMask':md_mask,
                'maskThresh':args.maskThresh,
                'phased':args.phased,
                'phaseError':args.phaseError,
                'MspDemographics':MspD,
                'seed':args.seed
                  }

    else:
        dg_params = {'N':nSamps,
            'Ne':Ne,
            'priorLowsRho':0.0,
            'priorHighsRho':rhoHi,
            'priorLowsMu':assumedMu * 0.66,
            'priorHighsMu':assumedMu * 1.33,
            'ChromosomeLength':maxLen,
            'winMasks':win_masks,
            'mdMask':md_mask,
            'maskThresh':args.maskThresh,
            'phased':args.phased,
            'phaseError':args.phaseError,
            'seed':args.seed
                  }


    # Assign pars for each simulation
    dg_train = Simulator(**dg_params)
    dg_vali = Simulator(**dg_params)
    dg_test = Simulator(**dg_params)


    ## Dump simulation pars for use with parametric bootstrap
    simParsFILE=os.path.join(networkDir,"simPars.p")
    with open(simParsFILE, "wb") as fOUT:
        dg_params["bn"]=os.path.basename(args.vcf).replace(".vcf","")
        pickle.dump(dg_params,fOUT)


    ## Simulate data
    print("Training set:")
    dg_train.simulateAndProduceTrees(numReps=args.nTrain,direc=trainDir,simulator="msprime",nProc=nProc)
    print("Validation set:")
    dg_vali.simulateAndProduceTrees(numReps=args.nVali,direc=valiDir,simulator="msprime",nProc=nProc)
    print("Test set:")
    dg_test.simulateAndProduceTrees(numReps=args.nTest,direc=testDir,simulator="msprime",nProc=nProc)
    print("\nSIMULATIONS FINISHED!\n")


    ## Count number of segregating sites in simulation
    SS=[]
    maxSegSites = 0
    minSegSites = float("inf")
    for ds in [trainDir,valiDir,testDir]:
        DsInfoDir = pickle.load(open(os.path.join(ds,"info.p"),"rb"))
        SS.extend(DsInfoDir["segSites"])
        segSitesInDs = max(DsInfoDir["segSites"])
        segSitesInDsMin = min(DsInfoDir["segSites"])
        maxSegSites = max(maxSegSites,segSitesInDs)
        minSegSites = min(minSegSites,segSitesInDsMin)


    ## Compare counts of segregating sites between simulations and input VCF
    print("SANITY CHECK")
    print("====================")
    print("numSegSites\t\t\tMin\tMean\tMax")
    print("Simulated:\t\t\t%s\t%s\t%s" %(minSegSites, int(sum(SS)/float(len(SS))), maxSegSites))
    for i in range(len(wins)):
        print("InputVCF %s:\t\t%s\t%s\t%s" %(wins[i][0],wins[i][3],wins[i][4],wins[i][5]))
    print("\n\n***ReLERNN_SIMULATE.py FINISHED!***\n")



def train_rel(args):
    
    
    ## Set seed
    if args.seed:
        os.environ['PYTHONHASHSEED']=str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    
    ## Set number of cores
    nProc = args.nCPU_tr
    
    
    ## Set up the directory structure to store the simulations data.
    if not args.outDir:
        print("Warning: No project directory found, using current working directory.")
        projectDir = os.getcwd()
    else:
        projectDir = args.outDir
    trainDir = os.path.join(projectDir,"train")
    valiDir = os.path.join(projectDir,"vali")
    testDir = os.path.join(projectDir,"test")
    networkDir = os.path.join(projectDir,"networks")
    model_trainedDir = os.path.join(networkDir,"pre_model")


#set transfer model parameter:

    if args.trans_flag:
        pretrained_model=os.path.join(model_trainedDir,"model.json")
        pretrained_weights = os.path.join(model_trainedDir,"weights.h5")
        if args.layer_fix_ind == None :
            print("you set transfer flag but don t put index layers list to freeze layers. enter this list:")
            l_ind=input()
            args.layer_fix_ind=list(map(int,(l_ind.split(','))))
          
            
    else:
        pretrained_model=None
        pretrained_weights = None


    ## Define output files
    test_resultFile = os.path.join(networkDir,"testResults.p")
    test_resultFig = os.path.join(networkDir,"testResults.pdf")
    modelSave = os.path.join(networkDir,"model.json")
    weightsSave = os.path.join(networkDir,"weights.h5")


    ## Identify padding required
    maxSimS = 0
    winFILE=os.path.join(networkDir,"windowSizes.txt")
    with open(winFILE, "r") as fIN:
        for line in fIN:
            maxSimS=max([maxSimS, int(line.split()[5])])
            print('_'*100)
            print('maxSimS:',maxSimS)
            print('_'*100)
    maxSegSites = 0
    for ds in [trainDir,valiDir,testDir]:
        DsInfoDir = pickle.load(open(os.path.join(ds,"info.p"),"rb"))
        segSitesInDs = max(DsInfoDir["segSites"])
        maxSegSites = max(maxSegSites,segSitesInDs)
        print('_'*100)
        print('segSitesInDs in for:{},maxSegSites in for:{}'.format(segSitesInDs,maxSegSites))
        print('_'*100)

    maxSegSites = max(maxSegSites, maxSimS)
    if args.trans_flag:
        jsonFILE = open(pretrained_model,"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        md=model_from_json(loadedModel)
        maxSegSites=int(md.layers[0].__dict__['_batch_input_shape'][1])-10
    print('_'*100)
    print('maxSegSites after for:{}'.format(maxSegSites))
    print('_'*100)

    
    ## Set network parameters
    bds_train_params = {
        'treesDirectory':trainDir,
        'targetNormalization':"zscore",
        'batchSize': 64,
        'maxLen': maxSegSites,
        'frameWidth': 5,
        'shuffleInds':True,
        'sortInds':False,
        'center':False,
        'ancVal':-1,
        'padVal':0,
        'derVal':1,
        'realLinePos':True,
        'posPadVal':0,
        'seqD':None,
        'seed':args.seed
              }


    ## Dump batch pars for bootstrap
    batchParsFILE=os.path.join(networkDir,"batchPars.p")
    with open(batchParsFILE, "wb") as fOUT:
        pickle.dump(bds_train_params,fOUT)


    bds_vali_params = copy.deepcopy(bds_train_params)
    bds_vali_params['treesDirectory'] = valiDir
    bds_vali_params['batchSize'] = 64

    bds_test_params = copy.deepcopy(bds_train_params)
    bds_test_params['treesDirectory'] = testDir
    DsInfoDir = pickle.load(open(os.path.join(testDir,"info.p"),"rb"))
    bds_test_params['batchSize'] = DsInfoDir["numReps"]
    bds_test_params['shuffleExamples'] = False


    ## Define sequence batch generator
    train_sequence = SequenceBatchGenerator(**bds_train_params)
    vali_sequence = SequenceBatchGenerator(**bds_vali_params)
    test_sequence = SequenceBatchGenerator(**bds_test_params)


    ## Train network
    runModels(ModelFuncPointer=GRU_TUNED84,
            ModelName="GRU_TUNED84",
            TrainDir=trainDir,
            TrainGenerator=train_sequence,
            ValidationGenerator=vali_sequence,
            TestGenerator=test_sequence,
            resultsFile=test_resultFile,
            network=[modelSave,weightsSave],
            numEpochs=args.nEpochs,
            validationSteps=args.nValSteps,
            nCPU=nProc,
            gpuID=args.gpuID,
            trans_flag=args.trans_flag,
            pretrained_network=[pretrained_model,pretrained_weights],
            layer_fix_ind=args.layer_fix_ind)


    ## Plot results of predictions on test set
    plotResults(resultsFile=test_resultFile,saveas=test_resultFig)


    print("\n***ReLERNN_TRAIN.py FINISHED!***\n")

'''
this is for test

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--vcf',dest='vcf',help='Filtered and QC-checked VCF file. Important: Every row must correspond to a biallelic SNP')
    parser.add_argument('-g','--genome',dest='genome',help='BED-formatted (i.e. zero-based) file corresponding to chromosomes and positions to evaluate')
    parser.add_argument('-m','--mask',dest='mask',help='BED-formatted file corresponding to inaccessible bases', default=None)
    parser.add_argument('-d','--projectDir',dest='outDir',help='Directory for all project output. NOTE: the same projectDir must be used for all functions of ReLERNN',default=None)
    parser.add_argument('-n','--demographicHistory',dest='dem',help='Output file from either stairwayplot, SMC++, or MSMC',default=None)
    parser.add_argument('-u','--assumedMu',dest='mu',help='Assumed per-base mutation rate',type=float,default=1e-8)
    parser.add_argument('-l','--assumedGenTime',dest='genTime',help='Assumed generation time (in years)',type=float)
    parser.add_argument('-r','--upperRhoThetaRatio',dest='upRTR',help='Assumed upper bound for the ratio of rho to theta',type=float,default=1.0)
    parser.add_argument('-t','--nCPU_sim',dest='nCPU_sim',help='Number of CPUs to use',type=int,default=None)
    parser.add_argument('-tt','--nCPU_tr',dest='nCPU_tr',help='Number of CPUs to use',type=int,default=1)
    parser.add_argument('-s','--seed',dest='seed',help='Random seed',type=int,default=None)
    parser.add_argument('--phased',help='Treat genotypes as phased',default=False, action='store_true')
    parser.add_argument('--unphased',dest='phased',help='Treat genotypes as unphased',action='store_false')
    parser.add_argument('--forceDiploid',help='Treat all samples as diploids with missing data (bad idea; see README)',default=False, action='store_true')
    parser.add_argument('--phaseError',dest='phaseError',help='Fraction of bases simulated with incorrect phasing',type=float,default=0.0)
    parser.add_argument('--maxSites',dest='winSizeMx',help='Max number of sites per window to train on. Important: too many sites causes problems in training (see README)!',type=int,default=1750)
    parser.add_argument('--forceWinSize',dest='forceWinSize',help='USED ONLY FOR TESTING, LEAVE AS DEFAULT',type=int,default=0)
    parser.add_argument('--maskThresh',dest='maskThresh',help='Discard windows where >= maskThresh percent of sites are inaccessible',type=float,default=1.0)
    parser.add_argument('--nTrain',dest='nTrain',help='Number of training examples to simulate',type=int,default=100000)
    parser.add_argument('--nVali',dest='nVali',help='Number of validation examples to simulate',type=int,default=1000)
    parser.add_argument('--nTest',dest='nTest',help='Number of test examples to simulate',type=int,default=1000)
    
    parser.add_argument('--nEpochs',dest='nEpochs',help='Maximum number of epochs to train (EarlyStopping is implemented for validation accuracy)', type=int, default=1000)
    parser.add_argument('--nValSteps',dest='nValSteps',help='Number of validation steps', type=int, default=20)
    parser.add_argument('--gpuID',dest='gpuID',help='Identifier specifying which GPU to use', type=int, default=0)
    parser.add_argument('--trans_flag',dest='trans_flag',help='if want use pre trained transfer model', type=bool, default=False)
    parser.add_argument('--layer_fix_ind',dest='layer_fix_ind',help='if want use pre trained transfer model give index of layer want to freeze in list format', type=list, default=None)



    args = parser.parse_args()

    #run simulate phase
    pr("start simulate phase")
    simualte_rel(args)

    #run train phase
    pr("start train phase")
    train_rel(args)



if __name__ == "__main__":
    main()
