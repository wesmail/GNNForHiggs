import sys
import argparse

import ROOT
import numpy as np
import polars as pl
from tqdm import tqdm

ROOT.gSystem.Load("Delphes-3.5.0/build/libDelphes.so")

try:
    ROOT.gInterpreter.Declare('#include "Delphes-3.5.0/classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare('#include "Delphes-3.5.0/external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=False, help="input root file", default="sig1.root")
    parser.add_argument('-o', type=str, required=False, help="output ascii file", default="signal")
    parser.add_argument('-n', type=int, required=False, help="number of events", default=100)
    args = parser.parse_args()

    inputFile = args.f
    # Create chain of root trees
    chain = ROOT.TChain("Delphes")
    chain.Add(inputFile)

    # Create object of class ExRootTreeReader
    treeReader = ROOT.ExRootTreeReader(chain)
    numberOfEntries = treeReader.GetEntries()
    print(f"Total number of events {numberOfEntries}")

    # Get pointers to branches used in this analysis
    branchJet = treeReader.UseBranch("Jet")
    branchElectron = treeReader.UseBranch("Electron")
    branchMuon = treeReader.UseBranch("Muon")
    branchMET = treeReader.UseBranch("MissingET")
    branchPhoton = treeReader.UseBranch("Photon")

    # ---------------------------------------------------------------
    # histograms
    # ---------------------------------------------------------------
    outHistFile = ROOT.TFile.Open("histos.root", "recreate")
    h01 = ROOT.TH1F("hMultiplicityJets", ";multiplicity; counts [a.u.]", 20, 0, 20)
    h02 = ROOT.TH1F("hMultiplicityMET", ";multiplicity; counts [a.u.]", 20, 0, 20)
    h03 = ROOT.TH1F("hMultiplicityElectron", ";multiplicity; counts [a.u.]", 20, 0, 20)
    h04 = ROOT.TH1F("hMultiplicityMuon", ";multiplicity; counts [a.u.]", 20, 0, 20)
    h05 = ROOT.TH1F("hMultiplicityPhoton", ";multiplicity; counts [a.u.]", 20, 0, 20)
    h06 = ROOT.TH1F("hMultiplicityBJets", ";multiplicity; counts [a.u.]", 20, 0, 20)
    # --------------------------------------------------------------- 
    nEvents = args.n
    if nEvents == 0 or nEvents > numberOfEntries:
        nEvents = numberOfEntries

    # feature vector xi
    # xi = (I1, I2, Il, Ib, pT , E, m) + (eta, phi) + (event_id)
    # I1 = ie electron "charge" [1, 0, -1]
    # I2 = is muon     "charge" [1, 0, -1]
    # Il 1 leading lepton, -1 2nd leading, 0 otherwise
    # Ib 1 leading jet, -1 2nd leading, 0 otherwise 
    
    event_data = [] 
    # Loop over all events
    for entry in tqdm(range(0, nEvents)):
        # Load selected branches with data from specified event
        treeReader.ReadEntry(entry)

        h01.Fill(branchJet.GetEntries())
        h02.Fill(branchMET.GetEntries())
        h03.Fill(branchElectron.GetEntries())
        h04.Fill(branchMuon.GetEntries())
        h05.Fill(branchPhoton.GetEntries())
        
        btagged_jets = []
        for j in range(branchJet.GetEntries()):
            jet_cand = branchJet.At(j)
            if (jet_cand.BTag == 1):
                btagged_jets.append(jet_cand)
                
        h06.Fill(len(btagged_jets))

        # require at least 2 b-jets
        if len(btagged_jets) < 2:
            continue

        for ind, obj in enumerate(btagged_jets):
            jet = ROOT.TLorentzVector()
            jet.SetPtEtaPhiM(obj.PT, obj.Eta, obj.Phi, obj.Mass)
            event_data.append([0, 0, 0, 1 if ind==0 else -1 if ind==1 else 0,
                                jet.Pt(), jet.E(), jet.M(), jet.Eta(), jet.Phi(), entry])      

        if branchElectron.GetEntries() > 1:
            # leptons (electrons)
            for ind, obj in enumerate(branchElectron):
                cand = ROOT.TLorentzVector()
                cand.SetPtEtaPhiM(obj.PT, obj.Eta, obj.Phi, 0.511*1e-3)
                event_data.append([obj.Charge, 0, 1 if ind==0 else -1 if ind==1 else 0, 0,
                                   cand.Pt(), cand.E(), cand.M(), cand.Eta(), cand.Phi(), entry])

        else:
            if branchMuon.GetEntries() > 1:
                # leptons (muons)
                for ind, obj in enumerate(branchMuon):
                    cand = ROOT.TLorentzVector()
                    cand.SetPtEtaPhiM(obj.PT, obj.Eta, obj.Phi, 0.511*1e-3)
                    event_data.append([0, obj.Charge, 1 if ind==0 else -1 if ind==1 else 0, 0,
                                    cand.Pt(), cand.E(), cand.M(), cand.Eta(), cand.Phi(), entry])

    x = np.vstack(event_data)
    df = pl.DataFrame(
        {
            "I1"      : x[:,0],
            "I2"      : x[:,1],
            "Il"      : x[:,2],
            "Ib"      : x[:,3],
            "Pt"      : x[:,4],
            "E"       : x[:,5],
            "M"       : x[:,6],
            "Eta"     : x[:,7],
            "Phi"     : x[:,8],
            "event_id": x[:, 9]
        }
    )
    df.write_csv("ascii_"+args.f.split('.')[0]+".csv") 

    outHistFile.cd()
    h01.Write()
    h02.Write()
    h03.Write()
    h04.Write()
    h05.Write()
    h06.Write()
    outHistFile.Close()

if __name__ == "__main__":
    main(sys.argv[1:])