Attribute VB_Name = "AutocallPricer"
'====================================================================
' AutocallPricer.bas
' ===================
' Version VBA du pricer Monte Carlo pour la note autocall a memoire.
' Meme mecanique EXACTE que python/pricer.py (memoire de coupon,
' autocall, protection du capital), 
' pense bete, pour etre importee dans le classeur autocall_pricer.xlsx 
' (menu VBA : clic droit sur le projet > Importer un fichier > AutocallPricer.bas) 
' et executee directement depuis Excel via le bouton 
' "Lancer le pricing" (a créer sur la feuille Pricing Summary, 
' Développeur > Insérer > Bouton, assigner la macro RunPricing).
'====================================================================

Option Explicit

' --- Genere une variable N(0,1) par la methode Box-Muller ---
Function RandNormal() As Double
    Dim u1 As Double, u2 As Double
    u1 = Rnd()
    u2 = Rnd()
    If u1 < 0.0000001 Then u1 = 0.0000001   ' evite Log(0)
    RandNormal = Sqr(-2 * Log(u1)) * Cos(2 * 3.14159265358979 * u2)
End Function

' --- Simule le prix a chaque date d'observation pour UNE trajectoire ---
Sub SimulatePath(S0 As Double, r As Double, q As Double, sigma As Double, _
                  obsTimes() As Double, ByRef S() As Double)
    Dim i As Integer, n As Integer
    Dim tPrev As Double, dt As Double, z As Double
    Dim logS As Double

    n = UBound(obsTimes)
    logS = Log(S0)
    tPrev = 0

    For i = 1 To n
        dt = obsTimes(i) - tPrev
        z = RandNormal()
        logS = logS + (r - q - 0.5 * sigma * sigma) * dt + sigma * Sqr(dt) * z
        S(i) = Exp(logS)
        tPrev = obsTimes(i)
    Next i
End Sub

' --- Valorise UNE trajectoire deja simulee et retourne le flux actualise total ---
Function PayoffPath(S() As Double, obsTimes() As Double, S0 As Double, notional As Double, _
                     autocallBarrier As Double, couponBarrier As Double, capitalBarrier As Double, _
                     couponRate As Double, r As Double) As Double
    Dim i As Integer, n As Integer
    Dim t As Double, period As Double, tPrev As Double
    Dim unpaidCoupon As Double, totalCashflow As Double
    Dim couponAmount As Double

    n = UBound(obsTimes)
    unpaidCoupon = 0
    totalCashflow = 0
    tPrev = 0

    For i = 1 To n
        t = obsTimes(i)
        period = t - tPrev
        tPrev = t

        If S(i) >= couponBarrier * S0 Then
            couponAmount = couponRate * period * notional + unpaidCoupon
            totalCashflow = totalCashflow + couponAmount * Exp(-r * t)
            unpaidCoupon = 0
        Else
            unpaidCoupon = unpaidCoupon + couponRate * period * notional
        End If

        If i < n Then
            ' Date d'observation intermediaire : verifie l'autocall
            If S(i) >= autocallBarrier * S0 Then
                totalCashflow = totalCashflow + notional * Exp(-r * t)
                PayoffPath = totalCashflow
                Exit Function   ' remboursement anticipe : la trajectoire s'arrete ici
            End If
        Else
            ' Derniere date : remboursement final (capital protege ou a risque)
            If S(i) >= capitalBarrier * S0 Then
                totalCashflow = totalCashflow + notional * Exp(-r * t)
            Else
                totalCashflow = totalCashflow + notional * (S(i) / S0) * Exp(-r * t)
            End If
        End If
    Next i

    PayoffPath = totalCashflow
End Function

' --- Point d'entree : lit les parametres depuis la feuille Inputs, lance le Monte Carlo, ecrit le resultat ---
Sub RunPricing()
    Dim wsInputs As Worksheet, wsSummary As Worksheet
    Dim S0 As Double, notional As Double, r As Double, q As Double, sigma As Double
    Dim autocallBarrier As Double, couponBarrier As Double, capitalBarrier As Double, couponRate As Double
    Dim obsTimes(1 To 3) As Double
    Dim S(1 To 3) As Double
    Dim nPaths As Long, i As Long
    Dim sumPayoff As Double, sumSqPayoff As Double, payoff As Double
    Dim price As Double, stderrValue As Double

    Set wsInputs = ThisWorkbook.Sheets("Inputs")
    Set wsSummary = ThisWorkbook.Sheets("Pricing Summary")

    S0 = wsInputs.Range("C5").Value
    notional = wsInputs.Range("C6").Value
    r = wsInputs.Range("C7").Value
    q = wsInputs.Range("C8").Value
    sigma = wsInputs.Range("C9").Value
    autocallBarrier = wsInputs.Range("C10").Value
    couponBarrier = wsInputs.Range("C11").Value
    capitalBarrier = wsInputs.Range("C12").Value
    couponRate = wsInputs.Range("C13").Value

    obsTimes(1) = wsInputs.Range("C16").Value
    obsTimes(2) = wsInputs.Range("C17").Value
    obsTimes(3) = wsInputs.Range("C18").Value

    nPaths = 20000   ' compromis vitesse/precision raisonnable pour une macro Excel (pas de vectorisation numpy ici)

    Randomize
    sumPayoff = 0
    sumSqPayoff = 0

    For i = 1 To nPaths
        Call SimulatePath(S0, r, q, sigma, obsTimes, S)
        payoff = PayoffPath(S, obsTimes, S0, notional, autocallBarrier, couponBarrier, capitalBarrier, couponRate, r)
        sumPayoff = sumPayoff + payoff
        sumSqPayoff = sumSqPayoff + payoff * payoff
    Next i

    price = sumPayoff / nPaths
    stderrValue = Sqr((sumSqPayoff / nPaths - price * price) / nPaths)

    wsSummary.Range("C7").Value = price
    wsSummary.Range("C8").Value = price / notional
    wsSummary.Range("C9").Value = stderrValue

    MsgBox "Pricing termine : " & Format(price, "0.00") & " (" & Format(price / notional, "0.00%") & _
           " du notional), erreur standard " & Format(stderrValue, "0.00"), vbInformation
End Sub
