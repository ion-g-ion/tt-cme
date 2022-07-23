from filecmp import cmpfiles
import torch as tn
import TTCME

tn.set_default_tensor_type(tn.DoubleTensor)

basis = [TTCME.basis.BSplineBasis(32,[0,1],2), TTCME.basis.BSplineBasis(43,[0,1],2)]
pdf = TTCME.pdf.pdfTT(basis, variable_names=['x1','x2'] ) 

pdf_beta = TTCME.pdf.BetaPdfTT([64,64],[2,2],[5,2])

exp = pdf_beta.expected_value()

basis2 = [TTCME.basis.BSplineBasis(64,[0,1],2), TTCME.basis.BSplineBasis(64,[0,1],2),TTCME.basis.BSplineBasis(32,[0,1],2), TTCME.basis.BSplineBasis(43,[0,1],2)]

pdf_c = TTCME.pdf.pdfTT.interpoalte(pdf = lambda x: x[...,0]**(2-1)*(1-x[...,0])**(x[...,2]-1) * x[...,1]**(2-1)*(1-x[...,1])**(x[...,3]-1), basis=[TTCME.basis.BSplineBasis(64,[0,1],2), TTCME.basis.BSplineBasis(64,[0,1],2)], basis_conditioned=[TTCME.basis.BSplineBasis(32,[2,3],2), TTCME.basis.BSplineBasis(43,[2, 3],2)], variable_names=['x1','x2'], conditioned_variable_names=['beta1','beta2'] )



