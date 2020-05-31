def classify(obj):   # Columns -> obj[0]: 0, obj[1]: 1, obj[2]: 2, obj[3]: 3, obj[4]: 4, obj[5]: 5, obj[6]: 6, obj[7]: 7, obj[8]: 8, obj[9]: 9, obj[10]: 10, obj[11]: 11, obj[12]: 12
    if float(obj[5]) < 0.6403294691885297:
        if float(obj[12]) < 0.37792998477929984:
            if float(obj[7]) < 0.018132294067049658:
                return '50.0'
            elif float(obj[7]) >= 0.018132294067049658:
                if float(obj[12]) < 0.25403348554033484:
                    if float(obj[9]) < 0.07552581261950289:
                        return '32.6'
                    elif float(obj[9]) >= 0.07552581261950289:
                        if float(obj[12]) < 0.10898021308980213:
                            return '28.860000000000003'
                        elif float(obj[12]) >= 0.10898021308980213:
                            if float(obj[5]) < 0.45851128737034785:
                                return '21.73333333333333'
                            elif float(obj[5]) >= 0.45851128737034785:
                                if float(obj[11]) < 0.9840889497439019:
                                    return '26.0'
                                elif float(obj[11]) >= 0.9840889497439019:
                                    if float(obj[8]) < 0.043478260869565216:
                                        return '22.433333333333334'
                                    elif float(obj[8]) >= 0.043478260869565216:
                                        return '24.338461538461537'
                elif float(obj[12]) >= 0.25403348554033484:
                    if float(obj[2]) < 0.21277777777777773:
                        return '18.099999999999998'
                    elif float(obj[2]) >= 0.21277777777777773:
                        if float(obj[6]) < 0.7319148936170212:
                            return '22.1125'
                        elif float(obj[6]) >= 0.7319148936170212:
                            return '20.07'
        elif float(obj[12]) >= 0.37792998477929984:
            if float(obj[0]) < 0.06484434127533403:
                if float(obj[6]) < 0.9356382978723403:
                    if float(obj[0]) < 0.0023618837022942934:
                        return '20.185714285714283'
                    elif float(obj[0]) >= 0.0023618837022942934:
                        return '17.4375'
                elif float(obj[6]) >= 0.9356382978723403:
                    return '14.924999999999999'
            elif float(obj[0]) >= 0.06484434127533403:
                if float(obj[0]) < 0.12867039278962275:
                    if float(obj[3]) < 0.5:
                        if float(obj[12]) < 0.7098934550989344:
                            return '12.3'
                        elif float(obj[12]) >= 0.7098934550989344:
                            return '15.5'
                    elif float(obj[3]) >= 0.5:
                        return '17.8'
                elif float(obj[0]) >= 0.12867039278962275:
                    return '9.630769230769232'
    elif float(obj[5]) >= 0.6403294691885297:
        if float(obj[11]) < 0.94909731730818:
            return '26.799999999999997'
        elif float(obj[11]) >= 0.94909731730818:
            if float(obj[1]) < 0.22222222222222224:
                return '45.53333333333333'
            elif float(obj[1]) >= 0.22222222222222224:
                return '39.769999999999996'
