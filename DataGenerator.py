import sys
import argparse

import ROOT
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT.gSystem.Load("libDelphes")

try:
    ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare(
        '#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=False,
                        help="input root file", default="../rootFiles/higgs.root")
    parser.add_argument('-o', type=str, required=True,
                        help="output ascii file", default="background")
    args = parser.parse_args()

    inputFile = args.f
    # Create chain of root trees
    chain = ROOT.TChain("Delphes")
    chain.Add(inputFile)

    # Create object of class ExRootTreeReader
    treeReader = ROOT.ExRootTreeReader(chain)
    numberOfEntries = treeReader.GetEntries()

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
    h01 = ROOT.TH1F("hMultiplicityJets",
                    ";multiplicity; counts [a.u.]", 20, 0, 20)
    h02 = ROOT.TH1F("hMultiplicityMET",
                    ";multiplicity; counts [a.u.]", 20, 0, 20)
    h03 = ROOT.TH1F("hMultiplicityElectron",
                    ";multiplicity; counts [a.u.]", 20, 0, 20)
    h04 = ROOT.TH1F("hMultiplicityMuon",
                    ";multiplicity; counts [a.u.]", 20, 0, 20)
    h05 = ROOT.TH1F("hMultiplicityPhoton",
                    ";multiplicity; counts [a.u.]", 20, 0, 20)
    # ---------------------------------------------------------------
    # feature vector xi (https://arxiv.org/pdf/1901.05627.pdf)
    # xi = (I1, I2, I3, I4, pT , E, m) + (eta, phi) + (event_id)
    # I1 = photon (1) or not 0
    # I2 = lepton (charge) or not 0
    # I3 = b-jet (1), light-jet (-1) or not a jet (0)
    # I4 = MET (1) or not (0)
    data = []
    # Loop over all events
    for entry in tqdm(range(0, numberOfEntries)):
        # Load selected branches with data from specified event
        treeReader.ReadEntry(entry)

        h01.Fill(branchJet.GetEntries())
        h02.Fill(branchMET.GetEntries())
        h03.Fill(branchElectron.GetEntries())
        h04.Fill(branchMuon.GetEntries())
        h05.Fill(branchPhoton.GetEntries())

        # photons
        for object in branchPhoton:
            photon = ROOT.TLorentzVector()
            photon.SetPtEtaPhiE(object.PT, object.Eta, object.Phi, object.E)
            data.append([1, 0, 0, 0, photon.Pt(), photon.E(),
                        photon.M(), photon.Eta(), photon.Phi(), entry])

        # leptons (electrons)
        for object in branchElectron:
            electron = ROOT.TLorentzVector()
            electron.SetPtEtaPhiM(object.PT, object.Eta,
                                  object.Phi, 0.511*1e-3)
            data.append([0, object.Charge, 0, 0, electron.Pt(
            ), electron.E(), electron.M(), electron.Eta(), electron.Phi(), entry])

        # leptons (muons)
        for object in branchMuon:
            muon = ROOT.TLorentzVector()
            muon.SetPtEtaPhiM(object.PT, object.Eta, object.Phi, 105.658*1e-3)
            data.append([0, object.Charge, 0, 0, muon.Pt(),
                        muon.E(), muon.M(), muon.Eta(), muon.Phi(), entry])

        # jets
        for object in branchJet:
            jet = ROOT.TLorentzVector()
            jet.SetPtEtaPhiM(object.PT, object.Eta, object.Phi, object.Mass)
            data.append([0, 0, 1 if object.BTag else -1, 0, jet.Pt(),
                        jet.E(), jet.M(), jet.Eta(), jet.Phi(), entry])

        # mets
        for object in branchMET:
            jet = ROOT.TLorentzVector()
            data.append([0, 0, 0, 1, object.MET, object.Eta,
                        object.Phi, object.Eta, object.Phi, entry])

    df = pd.DataFrame(
        data, columns=['I1', 'I2', 'I3', 'I4', 'Pt', 'E', 'M', 'Eta', 'Phi', 'Id'])

    print(df.head(15))
    df.to_csv("data_ascii_"+args.o+".csv")

    outHistFile.cd()
    h01.Write()
    h02.Write()
    h03.Write()
    h04.Write()
    h05.Write()
    outHistFile.Close()


if __name__ == "__main__":
    main(sys.argv[1:])
