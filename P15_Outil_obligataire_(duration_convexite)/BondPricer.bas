Attribute VB_Name = "BondPricer"
Option Explicit

' ============================================================
' BondPricer.bas
' ==============
' Outil de valorisation obligataire pour Excel, en VBA pur.
' Fonctions : prix, duration de Macaulay, duration modifiee,
' convexite, et interpolation lineaire d'une courbe de taux.
'
' Utilisation en cellule Excel :
'   =BondPrice(100, 0.05, 0.04, 2, 10)         -> prix
'   =MacaulayDuration(100, 0.05, 0.04, 2, 10)   -> duration (annees)
'   =ModifiedDuration(100, 0.05, 0.04, 2, 10)    -> duration modifiee
'   =BondConvexity(100, 0.05, 0.04, 2, 10)        -> convexite
' Macro RunBondPricing : lancee par le bouton "Lancer le pricing" du classeur
' bond_pricer.xlsm (feuille Results). Lit la feuille Inputs, ecrit Results.
' ============================================================

' Prix d'une obligation a coupons fixes.
' faceValue    : nominal
' couponRate   : taux de coupon annuel (ex: 0.05 = 5%)
' ytm          : taux de rendement actuariel annuel (yield to maturity)
' freq         : nombre de paiements de coupon par an (1, 2, 4, 12...)
' maturity     : maturite en annees
Function BondPrice(faceValue As Double, couponRate As Double, ytm As Double, _
                    freq As Integer, maturity As Double) As Double
    Dim n As Long, i As Long
    Dim couponPayment As Double, periodRate As Double
    Dim pv As Double

    n = CLng(maturity * freq)
    couponPayment = faceValue * couponRate / freq
    periodRate = ytm / freq

    pv = 0
    For i = 1 To n
        pv = pv + couponPayment / (1 + periodRate) ^ i
    Next i
    pv = pv + faceValue / (1 + periodRate) ^ n

    BondPrice = pv
End Function

' Duration de Macaulay : moyenne ponderee (par la valeur actuelle de
' chaque flux) des dates de paiement -- la "maturite effective" de
' l'obligation.
Function MacaulayDuration(faceValue As Double, couponRate As Double, ytm As Double, _
                           freq As Integer, maturity As Double) As Double
    Dim n As Long, i As Long
    Dim couponPayment As Double, periodRate As Double
    Dim price As Double, weightedSum As Double
    Dim cashflow As Double, t As Double

    n = CLng(maturity * freq)
    couponPayment = faceValue * couponRate / freq
    periodRate = ytm / freq

    price = BondPrice(faceValue, couponRate, ytm, freq, maturity)
    weightedSum = 0

    For i = 1 To n
        t = i / freq
        cashflow = couponPayment
        If i = n Then cashflow = cashflow + faceValue
        weightedSum = weightedSum + t * cashflow / (1 + periodRate) ^ i
    Next i

    MacaulayDuration = weightedSum / price
End Function

' Duration modifiee : sensibilite relative du prix a une variation du
' taux (approximation lineaire). ModifiedDuration = MacDuration / (1+y/freq)
Function ModifiedDuration(faceValue As Double, couponRate As Double, ytm As Double, _
                           freq As Integer, maturity As Double) As Double
    Dim macDur As Double
    macDur = MacaulayDuration(faceValue, couponRate, ytm, freq, maturity)
    ModifiedDuration = macDur / (1 + ytm / freq)
End Function

' Convexite : terme du second ordre de la sensibilite du prix au taux
' (la duration seule sous-estime la variation de prix pour un choc de
' taux important -- la convexite corrige cette approximation lineaire).
Function BondConvexity(faceValue As Double, couponRate As Double, ytm As Double, _
                        freq As Integer, maturity As Double) As Double
    Dim n As Long, i As Long
    Dim couponPayment As Double, periodRate As Double
    Dim price As Double, weightedSum As Double
    Dim cashflow As Double, t As Double

    n = CLng(maturity * freq)
    couponPayment = faceValue * couponRate / freq
    periodRate = ytm / freq

    price = BondPrice(faceValue, couponRate, ytm, freq, maturity)
    weightedSum = 0

    For i = 1 To n
        t = i / freq
        cashflow = couponPayment
        If i = n Then cashflow = cashflow + faceValue
        weightedSum = weightedSum + cashflow * t * (t + 1 / freq) / (1 + periodRate) ^ i
    Next i

    BondConvexity = weightedSum / (price * (1 + periodRate) ^ 2)
End Function

' Approximation du changement de prix pour un choc de taux donne,
' combinant duration (1er ordre) et convexite (2eme ordre) --
' l'usage pratique le plus courant de ces deux mesures pour un desk.
Function PriceChangeApprox(faceValue As Double, couponRate As Double, ytm As Double, _
                            freq As Integer, maturity As Double, yieldShock As Double) As Double
    Dim modDur As Double, convex As Double, price As Double

    modDur = ModifiedDuration(faceValue, couponRate, ytm, freq, maturity)
    convex = BondConvexity(faceValue, couponRate, ytm, freq, maturity)
    price = BondPrice(faceValue, couponRate, ytm, freq, maturity)

    PriceChangeApprox = price * (-modDur * yieldShock + 0.5 * convex * yieldShock ^ 2)
End Function

' Interpolation lineaire simple d'une courbe de taux (tableaux de
' maturites et de taux de meme taille), pour obtenir un taux a une
' maturite intermediaire non cotee directement.
Function InterpolateYield(maturities As Range, yields As Range, targetMaturity As Double) As Double
    Dim n As Long, i As Long
    n = maturities.Cells.Count

    If targetMaturity <= maturities.Cells(1).Value Then
        InterpolateYield = yields.Cells(1).Value
        Exit Function
    End If
    If targetMaturity >= maturities.Cells(n).Value Then
        InterpolateYield = yields.Cells(n).Value
        Exit Function
    End If

    For i = 1 To n - 1
        Dim t1 As Double, t2 As Double, y1 As Double, y2 As Double
        t1 = maturities.Cells(i).Value: t2 = maturities.Cells(i + 1).Value
        y1 = yields.Cells(i).Value: y2 = yields.Cells(i + 1).Value
        If targetMaturity >= t1 And targetMaturity <= t2 Then
            InterpolateYield = y1 + (y2 - y1) * (targetMaturity - t1) / (t2 - t1)
            Exit Function
        End If
    Next i
End Function

' ============================================================
' Macro principale, assignee au bouton "Lancer le pricing"
' Lit les parametres de la feuille Inputs et ecrit tout dans Results
' ============================================================
Sub RunBondPricing()
    Dim wi As Worksheet, wr As Worksheet
    Dim face As Double, cpn As Double, y As Double, mat As Double, shock As Double
    Dim freq As Integer, i As Long, s As Double
    Dim price As Double, shockedPrice As Double

    Set wi = ThisWorkbook.Worksheets("Inputs")
    Set wr = ThisWorkbook.Worksheets("Results")

    face = wi.Range("C5").Value
    cpn = wi.Range("C6").Value
    y = wi.Range("C7").Value
    freq = CInt(wi.Range("C8").Value)
    mat = wi.Range("C9").Value
    shock = wi.Range("C10").Value

    If freq <= 0 Or mat <= 0 Or face <= 0 Then
        MsgBox "Parametres invalides : nominal, frequence et maturite doivent etre > 0.", vbExclamation
        Exit Sub
    End If

    Application.ScreenUpdating = False

    price = BondPrice(face, cpn, y, freq, mat)
    shockedPrice = BondPrice(face, cpn, y + shock, freq, mat)

    wr.Range("C5").Value = price
    wr.Range("C6").Value = MacaulayDuration(face, cpn, y, freq, mat)
    wr.Range("C7").Value = ModifiedDuration(face, cpn, y, freq, mat)
    wr.Range("C8").Value = BondConvexity(face, cpn, y, freq, mat)
    wr.Range("C9").Value = PriceChangeApprox(face, cpn, y, freq, mat, shock)
    wr.Range("C10").Value = shockedPrice - price
    wr.Range("C11").Value = wr.Range("C9").Value - wr.Range("C10").Value
    wr.Range("C12").Value = InterpolateYield(wi.Range("B14:B18"), wi.Range("C14:C18"), wi.Range("C20").Value)

    ' Table de sensibilite : chocs de -200 a +200 bps par pas de 50 bps
    For i = 0 To 8
        s = (i - 4) * 0.005
        wr.Cells(17 + i, 2).Value = s
        wr.Cells(17 + i, 3).Value = BondPrice(face, cpn, y + s, freq, mat)
        wr.Cells(17 + i, 4).Value = wr.Cells(17 + i, 3).Value - price
        wr.Cells(17 + i, 5).Value = PriceChangeApprox(face, cpn, y, freq, mat, s)
        wr.Cells(17 + i, 6).Value = wr.Cells(17 + i, 5).Value - wr.Cells(17 + i, 4).Value
    Next i

    Application.ScreenUpdating = True
    wr.Activate
    MsgBox "Pricing termine. Prix = " & Format(price, "0.0000"), vbInformation
End Sub

