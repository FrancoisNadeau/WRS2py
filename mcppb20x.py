function mcppb20(formula, data, tr = 0.2, nboot = 599, ...){
    if (data === undefined) {
        var mf = model.frame(formula);
    } else {
        var mf = model.frame(formula, data);
    }
    var cl = match.call();
    var x = split(model.extract(mf, "response"), mf[, 2]);
    var con = 0;
    var alpha = 0.05;
    var grp = NaN;
    var WIN = false;
    var win = 0.1;
    var crit = NaN;
    con = Array.from(con);
    if (Array.isArray(x)) {
        var xx = [];
        for (var i = 0; i < x[0].length; i++) {
            xx[i] = x[i];
        }
        x = xx;
    }
    if (!Array.isArray(x)) throw new Error("Expected data in a list or array.");
    if (!isNaN(sum(grp))) {
        var xx = [];
        for (var i = 0; i < grp.length; i++) {
            xx[i] = x[grp[0]];
        }
        x = xx;
    }
    var J = x.length;
    var tempn = Array(J).fill(0);
    for (var j = 0; j < J; j++) {
        var temp = x[j];
        temp = temp.filter(function (val) { return !isNaN(val); });
        tempn[j] = temp.length;
        x[j] = temp;
    }
    var Jm = J - 1;
    var d = (sum(con.map(function (val) { return val * val; })) == 0) ? (J * J - J) / 2 : con[0].length;
    if (isNaN(crit) && tr != 0.2) {
        throw new Error("Must specify critical value if trimming amount differs from 0.2");
    }
    if (WIN) {
        if (tr < 0.2) {
            console.warn("When Winsorizing, the amount of trimming should be at least 0.2");
        }
        if (win > tr) throw new Error("Amount of Winsorizing must <= amount of trimming");
        if (Math.min(...tempn) < 15) {
            console.warn("Winsorizing samples of n < 15 lessens Type I error control.");
        }
        for (var j = 0; j < J; j++) {
            x[j] = winval(x[j], win);
        }
    }
    if (isNaN(crit)) {
        var alpha_dict = {
            "0.025": (d == 1) ? true : false,
            "0.0005": (d == 10 && alpha == 0.025 && nboot <= 2000) ? true : false,
            "0.0006": (d == 15 && alpha == 0.025 && nboot == 2000) ? true : false,
            "0.001": (d == 3 && alpha == 0.01 && [1000, 2000, 3000].includes(nboot)) ? true : false,
            "0.0015": (d == 6 && alpha == 0.025 && nboot == 2000) ? true : false,
            "0.0016": (d == 15 && alpha == 0.05 && nboot == 2000) ? true : false,
            "0.002": (d == 6 && alpha == 0.025 && nboot == 1000) || (d == 10 && alpha == 0.05 && nboot <= 2000) ? true : false,
            "0.0023": (d == 10 && alpha == 0.05 && nboot == 3000) ? true : false,
            "0.0026": (d == 15 && alpha == 0.05 && nboot == 5000) ? true : false,
            "0.004": (d == 3 && alpha == 0.025 && [1000, 2000].includes(nboot)) || (d == 6 && alpha == 0.05 && nboot == 1000) ? true : false,
            "0.0045": (d == 3 && alpha == 0.025 && nboot == 2000) ? true : false,
            "0.006": (d == 5 && alpha == 0.05 && nboot == 2000) ? true : false,
            "0.007": (d == 4 && alpha == 0.05 && nboot == 2000) ? true : false,
            "0.0085": (d == 3 && alpha == 0.05 && [1000, 2000].includes(nboot)) ? true : false,
            "0.014": (d == 2 && alpha == 0.05 && [1000, 2000].includes(nboot)) ? true : false
        };
        crit = Object.keys(alpha_dict).find(function (key) { return alpha_dict[key]; });
    }
    if (isNaN(crit) && alpha == 0.05) crit = 0.0268660714 * (1 / d) - 0.0003321429;
    if (isNaN(crit)) crit = alpha / (2 * d);
    if (d > 10 && nboot < 5000) {
        console.warn("Suggest using nboot = 5000 if more than 10 contrasts.");
    }
    var icl = Math.round(crit * nboot) + 1;
    var icu = Math.round((1 - crit) * nboot);
    if (sum(con.map(function (val) { return val * val; })) == 0) {
        con = Array(J).fill(Array(d).fill(0));
        var id = 0;
        for (var j = 0; j < Jm; j++) {
            var jp = j + 1;
            for (var k = jp; k < J; k++) {
                id++;
                con[j][id] = 1;
                con[k][id] = -1;
            }
        }
    }
    var psihat = Array.from(Array(con[0].length), function () { return Array(6).fill(0); });
    var psihat_dimnames = [null, [
        "con.num", "psihat", "se", "ci.lower",
        "ci.upper", "p-value"
    ]];
    psi_hat = psi_hat.map(function (val, index) { return val.map(function (v, i) { return psi_hat_dimnames[index][i]; }); });
    var bvec = Array(J).fill(Array(nboot).fill(NaN));
    for (var j = 0; j < J; j++) {
        var data = Array.from({ length: x[j].length * nboot }, function () { return sample(x[j], 1)[0]; });
        data = Array.from({ length: nboot }, function (_, i) { return data.slice(i * x[j].length, (i + 1) * x[j].length); });
        bvec[j] = data.map(function (val) { return mean(val, tr); });
    }
    var test = Array(nboot).fill(NaN);
    for (var d = 0; d < con[0].length; d++) {
        var top = 0;
        for (var i = 0; i < J; i++) {
            top += con[i][d] * bvec[i];
        }
        test[d] = (sum(top.map(function (val) { return val > 0 ? 1 : (val == 0 ? 0.5 : 0); })) / nboot);
        test[d] = Math.min(test[d], 1 - test[d]);
        top.sort();
        psi_hat[d][3] = top[icl];
        psi_hat[d][4] = top[icu];
    }
    for (var d = 0; d < con[0].length; d++) {
        psi_hat[d][0] = d;
        var testit = lincon1(x, con.map(function (val) { return val[d]; }), tr, false);
        psi_hat[d][5] = 2 * test[d];
        psi_hat[d][1] = testit.psihat[0][1];
        psi_hat[d][2] = testit.test[0][3];
    }
    return {
        psihat: psi_hat,
        crit_p_value: 2 * crit,
        con: con
    };
}


