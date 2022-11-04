#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <regex>
#include "Eigen\Dense"
#include "EigenRand\EigenRand"
#include <utility>

using namespace std;
using namespace Eigen;

class Layer
{
    /*
        Vispārīgā neironu tīkla slāņā klase.
        Satur izvades, ievades datus no forward propogation
        un savu atvasinājumu(dinputs) priekš backpropogation
    */

protected:
    Rand::Vmt19937_64 urng{ time(0) };
    MatrixXd dinputs;
    MatrixXd inputs;
    MatrixXd output;


public:
    virtual void forward(const MatrixXd& inputs, bool training) = 0; //Aprēķina slāņa rezultātus
    virtual void backward(const MatrixXd& dvalues) = 0; //Aprēķina slāņa atvasinājumus

    MatrixXd getOutput()const
    {
        return output;
    }

    MatrixXd getDinputs()const
    {
        return dinputs;
    }

    virtual string serialize() const = 0; // atgriež string, kas satur serializētu informāciji slāņa atkārtotai izveidei
    static vector<string> deserializeArgs(string serial)
    {
        /*
            Atgriež vektoru ar serializētā slāņa argumentiem
            Vairāk par serializētā slāņā pierakstu Model::saveNetwork
        */

        static const regex argsRegex("(\\(|\\,)((\\d|\\.)+)");
        smatch argsMatch;
        vector<string> args;
        while(regex_search(serial, argsMatch, argsRegex))
        {
            args.push_back(argsMatch.str(2));
            serial = argsMatch.suffix().str();
        }
        return args;
    }
};

class LayerDense : public Layer{

    /*
        Slānis, kas satur neironus ar to weights un bias
        Kā arī iekļauj neobligātu L1 un L2 regularizāciju
    */

    MatrixXd weights;
    RowVectorXd biases;
    MatrixXd dweights;
    MatrixXd dweightsSum;
    RowVectorXd dbiases;
    MatrixXd dbiasesSum;
    double L1;
    double L2;

    //Parametri optimizācijas funkcijām
    bool optimizationParamsSet = false;
    MatrixXd weightMomentums;
    RowVectorXd biasMomentums;
    MatrixXd weightCache;
    RowVectorXd biasCache;

    public:

    LayerDense(int nInput, int nNeurons, double L1=0, double L2=0)
        :L1(L1), L2(L2)
    {
        weights = Rand::balanced<MatrixXd>(nInput, nNeurons, urng); //svarus izvēlas no -1 līdz 1
        biases = RowVectorXd::Zero(nNeurons); //bias sākas visi kā nulles
        dweightsSum = MatrixXd::Zero(nInput, nNeurons);
        dbiasesSum = RowVectorXd::Zero(nNeurons);
    }

    void forward(const MatrixXd& inputs, bool training)
    {
        this->inputs = inputs; //saglabā ievadi priekš atvasinājuma aprēķināšanas
        this->output = (inputs * weights).rowwise() + biases; //Aprēķina katras ievades reizinājumu ar tai piemērotu svaru, un saskaita atsevišķo neironu rezultātus (algebriska matricu reizināšana)
    }                                                         //Pieskaita katra neironu rezultātam tā bias


    void backward(const MatrixXd& dvalues) //dvalues ir priekšējo slāņu atvasinājums
    {
        MatrixXd currDWeights = inputs.transpose() * dvalues; // Aprēķina d/dw2 f(g(x,w2,b2),w1,b1) = d/dx f(g(x,w2,b2),w1,b1) * d/dw2 g(x,w2,b2),
                               //kur f(x,w,b) = g(x,w,b) = xw + b , x - ievade, d/dx f(g(x,w2,b2),w1,b1)- dvalues
        RowVectorXd currDBiases = dvalues.colwise().sum(); //Līdzīgi aprēķina bias avasinājumus, d/db1 g(x,w2,b2) = 1, tātad to neraksta
                               //saskaita dvalues, jo viens bias ieteikmē vairākas vērtības

        //Aprēķina zaudējuma regularizācijas vērtības jau pašā slānī (pirms zaudējuma funckcijas)
        // L1 un L2 ir lambda konstantes, kas nosaka regularizācijas stiprumu
        // Zaudējuma atvasinājumus pieskaita dweights un dbiases

        if (L1 > 0)
        {
            MatrixXd dL1 = weights.array().sign(); // d/dx +-x*l = +-l
            currDWeights +=  dL1 * L1;
        }

        if (L2 > 0)
        {
             currDWeights += weights * (2 * L2); //d/dx lx^2 = 2xl
        }

        if (L1 > 0)
        {
            MatrixXd dL1 = biases.array().sign();
            currDBiases += dL1 * L1;
        }

        if(L2 > 0)
        {
            currDBiases += biases * (2 * L2);
        }

        dweightsSum += currDWeights;
        dbiasesSum += currDBiases;
        dinputs = dvalues * weights.transpose(); //Aprēķina d/dx no šī slāņa funkcijas ievadēm
    }


    void calcGradients(int batchSize)
    {
        /*
            Kad viens batch ir pabeigts aprēķina atvasinājumu vidējās vērt.
        */
        dweights = dweightsSum.array() / batchSize;
        dbiases = dbiasesSum.array() / batchSize;
        dweightsSum = MatrixXd::Zero(weights.rows(), weights.cols());
        dbiasesSum = RowVectorXd::Zero(biases.cols());
    }

    string serialize() const
    {
        string serialized = "LD(";
        serialized += to_string(weights.rows()) + ',';
        serialized += to_string(weights.cols()) + ',';
        serialized += to_string(L1) + ',';
        serialized += to_string(L2);
        serialized += ")";

        // Šī slāņa serializācija satur papildus datus, kas ir tā weights un biases
        //w(skaitlis) apzīmē svaru un b(skaitlis) bias
        //Tātad tā serializācija piem., LD(x,y,z,e)w4.3w5.5b2.0

        for(auto r = 0; r<weights.rows(); r++)
        {
            for(auto c = 0; c<weights.cols(); c++)
            {
                serialized+="w"+to_string(weights.coeff(r,c));
            }
        }
        for(auto i = 0; i<biases.cols(); i++)
        {
            serialized+="b"+to_string(biases.coeff(i));
        }
        return serialized;
    }

    static Layer* deserialize(string serial)
    {
        auto args = deserializeArgs(serial);
        LayerDense* ld = new LayerDense(stoi(args[0]), stoi(args[1]), stod(args[2]), stod(args[3]));

        // Ar regex iegūst LayerDense::Serialize() minēto weights un biases informāciju, ko ieliek jaunā objekta attiecīgajos std::vector

        static const regex weightsRegex("w((\\d|\\.|\\-)+)");
        smatch weightsMatch;
        int row = 0, col = 0;
        while(regex_search(serial, weightsMatch, weightsRegex))
        {
            ld->weights.coeffRef(row,col) = stod(weightsMatch.str(1));
            serial = weightsMatch.suffix().str();
            col++;
            if(col==ld->weights.cols())
            {
                col = 0;
                row++;
            }
        }
        static const regex biasesRegex("b((\\d|\\.|\\-)+)");
        smatch biasesMatch;
        int pos = 0;
        while(regex_search(serial, biasesMatch, biasesRegex))
        {
            ld->biases.coeffRef(pos) = stod(biasesMatch.str(1));
            serial = biasesMatch.suffix().str();
            pos++;
        }
        return (Layer*)ld;
    }

    void print() const
    {
        //testēšanai
        cout<<weights<<endl<<endl;
        cout<<biases<<endl;
    }

    //Optimizācijas klasēm ļauj piekļuvi privātajiem mainīgajiem
    //jo tās ir cieši piesaistītas klases

    friend class Adam;
};

class LayerDropout : public Layer
{
    /*
        Dropout klase ļauj izsviest daļu slāņa izvades rezultātu(nonullē), lai atsevišķi neironi nekļūtu pārāk stipri
    */
    double rate;
    MatrixXd binaryMask;

public:
    LayerDropout(double rate)
    {
        assert(rate<=1 && rate>=0);
        // Apgriež intuvitīvo izsviešanas procentu par ērtāko atstāšanas procentu
        this->rate = 1 - rate;
    }

    void forward(const MatrixXd& inputs, bool training)
    {
        this->inputs = inputs;
        if(!training) // Ja nenotiek trenēšana šo slāni izlaiž
        {
            output = inputs;
            return;
        }

        // Ģenerē nejaušu vieninieku un nullīšu matricu ar rate procentu vieninieku
        binaryMask = Rand::binomial<Matrix<int, -1, -1>>(inputs.rows(), inputs.cols(), urng, 1, rate).cast<double>();
        //piereizina masku attiecīgajiem ievades koeficentiem
        output = inputs.array() * binaryMask.array();
    }

    void backward(const MatrixXd& dvalues)
    {
        dinputs = dvalues.array() * binaryMask.array(); //Tādā pašā veidā masku izmanto ar atvasinājumiem
    }

    string serialize() const
    {
        string serialized = "DL(";
        serialized += to_string(1 - rate); //apgriež atpakaļ
        serialized += ")";
        return serialized;
    }

    static Layer* deserialize(string serial)
    {
        auto args = deserializeArgs(serial);
        Layer* dl = new LayerDropout(stod(args[0]));
        return dl;
    }
};

class Activation : public Layer
{
    /*
        Vispārīgā aktivizācijas klases
        Aktvizācija piemēro savu funkciju pēc Dense slāņa
        Darbojas kā viens no tīkla slāņiem
    */
public:
    virtual void forward(const MatrixXd& inputs, bool training) = 0;
    virtual void backward(const MatrixXd& dvalues) = 0;
};

class ActivationReLU : public Activation
{
    /*
    f(x,l) = {x >= 0 , x    x<0 ,  x*l} , kur l - leak
    */
public:

    double leak;

    ActivationReLU(double leak = 0)
    {
        assert(leak >= 0 && leak < 1);
        this->leak = leak;
    }

    void forward(const MatrixXd& inputs, bool training)
    {
        this->inputs = inputs;
        output = (inputs.array() <= 0).select(inputs * leak, inputs);
    }

    void backward(const MatrixXd& dvalues)
    {
        //f'(x) = {x >= 0 , 1    x<0 ,  l}

        dinputs = (inputs.array() <= 0).select(dvalues * leak, dvalues);
    }

    string serialize() const
    {
        string serialized = "RE(";
        serialized += to_string(leak);
        serialized += ")";
        return serialized;
    }

    static Layer* deserialize(string serial)
    {
        auto args = deserializeArgs(serial);
        Layer* relu = new ActivationReLU(stod(args[0]));
        return relu;
    }
};

class Sigmoid : public Activation
{
    /*
        f(x) = 1/(1+e^(-x))
    */
public:

    void forward(const MatrixXd& inputs, bool training)
    {
        this->inputs = inputs;
        auto one = ArrayXXd::Constant(inputs.rows(), inputs.cols(), 1);
        output = (one+(inputs*-1).array().exp()).pow(-1);
    }

    void backward(const MatrixXd& dvalues)
    {
        //f'(x) = f(x) * (1-f(x))
        auto one = MatrixXd::Constant(output.rows(), output.cols(), 1);
        dinputs = output.array() * (one-output).array() * dvalues.array();
    }

    string serialize() const
    {
        string serialized = "SI(";
        serialized += ")";
        return serialized;
    }

    static Layer* deserialize(string serial)
    {
        return new Sigmoid;
    }
};

class Loss
{
    /*
        Vispārīgā zaudējuma klase
        Zaudejuma klases pielieto trenējot neironu tīklu, lai noskaidrotu, cik nepareizs tas ir
        Nepareizuma vērtība, jeb zaudējums tiek saskaitīts un var iegūt vidējo vērtību statistikas nolūkos
    */

public:
    double accumulatedSum, accumulatedCount;

    double calculate(const MatrixXd& output, const MatrixXd& corr)
    {
        // Aprēķina output zaudējumu pret corr(pareizo) un to atgriež

        auto sampleLosses = forward(output, corr);
        auto dataLoss = sampleLosses.array().mean(); //zaudējums ir vidējā vērtība, ja ir vairākvērtību atbilde

        //Saglabā statistiku
        accumulatedSum += sampleLosses.array().sum();
        accumulatedCount += sampleLosses.cols();
        return dataLoss;
    }

    double calculateAccumulated() const
    {
        return accumulatedSum / accumulatedCount;
    }

    void newPass()
    {
        /*
            Atiestata vidējās vērtības statistiku
        */
        accumulatedSum = 0;
        accumulatedCount = 0;
    }

    virtual RowVectorXd forward(const MatrixXd& pred, const MatrixXd& corr)=0; //Atgriež zaudējumu katrai no atbildēm

    virtual MatrixXd backward(const MatrixXd& pred, const MatrixXd& corr)=0; //Atgriež katras ievades atvasinājumu

};

class LossMeanSquaredError : public Loss
{

// f(corr, pred) = 1/n * sigma[i->n ,(corr(i) - pred(i))^2];

public:
        RowVectorXd forward(const MatrixXd& pred, const MatrixXd& corr)
        {
            auto sampleLosses = (corr - pred).array().square().colwise().mean();
            return sampleLosses;
        }

        MatrixXd backward(const MatrixXd& pred, const MatrixXd& corr)
        {
            //d/dpred(i) f(corr, pred) = -2(corr - pred);

            MatrixXd dinputs = (corr - pred).array() * -2;
            return dinputs;
        }
};

class BinaryCrossEntropy : public Loss
{

// f(corr, pred) = -1/n * sigma[i->n ,(corr(i) * ln(pred(i)) + (1-corr)*ln(1-pred)];

    MatrixXd getClipped(MatrixXd input)
    {
        /*
            Atgriež input ar vērtībām, kur nuļļu un vieninieku vērtības ir nomainītas ar tuvām vērtībām ]0;1[
        */

        static const auto closeToZero = pow(10, -7);
        static const auto closeToOne = 1.0 - closeToZero;

        return input.cwiseMax(closeToZero).cwiseMin(closeToOne);
    }
public:
    RowVectorXd forward(const MatrixXd& pred, const MatrixXd& corr)
    {

        auto one = MatrixXd::Constant(pred.rows(), pred.cols(),1);

        auto clippedPred = getClipped(pred);     //Ievade tikai vērtības ]0;1[

        auto sampleLosses = (corr.array()*clippedPred.array().log() + (one-corr).array()*(one-clippedPred).array().log()).colwise().mean() * -1;
        return sampleLosses;
    }

    MatrixXd backward(const MatrixXd& pred, const MatrixXd& corr)
    {
        //Ievade tikai vērtības ]0;1[
        //d/dpred(i) f(corr, pred) = -(corr(i)/pred(i) - (1-corr)/(1-pred))
        auto one = MatrixXd::Constant(pred.rows(), pred.cols(),1);

        auto clippedPred = getClipped(pred);
         //Ievade tikai vērtības ]0;1[
        auto dinputs = (corr.array()/clippedPred.array() - (one-corr).array()/(one-clippedPred).array()) * -1;
        return dinputs;
    }
};

class Optimizer
{
    /*
        Vispārīgā optimizācijas klase
    */
public:
     virtual void preUpdateParams()=0; // Izsauc pirms piemēro optimizācijas gradient
     virtual void updateParams(LayerDense& layer) = 0; //Izsauc, lai piemērotu optimizāciju katram slāniem
     virtual void postUpdateParams() = 0; // Izsauc pec optimizācijas gradient piemērošanas
};

class Adam : public Optimizer
{
    double learningRate, decay, epsilon, beta1, beta2;
    int iterations = 0;
    double currentLearningRate;
public:
    Adam(double learningRate=0.001, double decay=0, double epsilon=0.0000001,
    double beta1=0.9, double beta2=0.999)
        :learningRate(learningRate), decay(decay), epsilon(epsilon), beta1(beta1), beta2(beta2)
    {
        currentLearningRate = learningRate;
    }

    void preUpdateParams()
    {
        /*
            Eksponenciāli samazina learning rate pēc katra postUpdateParams()
        */
        if(decay>0)
        {
            currentLearningRate = learningRate * exp(-decay*iterations);
        }

    }

    void updateParams(LayerDense& layer)
    {
        if(!layer.optimizationParamsSet)
        {
            /*
                Sākot optimizāciju inicializē nepieciešamos mainīgos
            */

            auto weightZeroes = MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
            auto biasZeroes = RowVectorXd::Zero(layer.biases.cols());
            layer.weightMomentums = weightZeroes;
            layer.weightCache = move(weightZeroes);
            layer.biasMomentums = biasZeroes;
            layer.biasCache = move(biasZeroes);
            layer.optimizationParamsSet = true;
        }

        /*
            Aprēķina momentum un cache pēc Adam algoritma nosacījumiem un piemēro correction pēc iteration(jeb parasti epoch skaita)
        */

        layer.weightMomentums = layer.weightMomentums.array() * beta1 +  (layer.dweights * (1 - beta1)).array();
        layer.biasMomentums = layer.biasMomentums.array() * beta1 +  (layer.dbiases * (1 - beta1)).array();

        MatrixXd weightMomentumsCorrected = layer.weightMomentums.array() / (1 - pow(beta1, (iterations + 1)));
        MatrixXd biasMomentumsCorrected = layer.biasMomentums.array() / (1 - pow(beta1, (iterations + 1)));

        layer.weightCache = layer.weightCache.array() * beta2 + layer.dweights.array().square() * (1 - beta2);
        layer.biasCache = layer.biasCache.array() * beta2 + layer.dbiases.array().square() * (1 - beta2);

        MatrixXd weightCacheCorrected = layer.weightCache.array() / (1 - pow(beta2, (iterations + 1)));
        MatrixXd biasCacheCorrected = layer.biasCache.array() / (1 - pow(beta2, (iterations + 1)));

        //Pēc standarta SGD principa atņem no weights un bias vērtībām slānī, bet nevis tiešā veidā atvasinājumus, bet
        //Adam algoritma ģenerēto momentum un cache aprēķinu

        layer.weights -= MatrixXd((weightMomentumsCorrected * currentLearningRate).array() /
        (weightCacheCorrected.array().sqrt() + epsilon));
        layer.biases -= MatrixXd((biasMomentumsCorrected * currentLearningRate).array() /
        (biasCacheCorrected.array().sqrt() + epsilon));

    }

    void postUpdateParams()
    {
        iterations += 1;
    }
};

class Model
{

/*
    Neironu tīkla modeļa klase, ko izmanto, lai apvienotu vairākus slāņus un vienkārši aprēķinātu to rezultātu un tos trenētu
*/

list<Layer*> layers;
list<LayerDense*> trainableLayers;
list<Layer*> loadedLayers;
default_random_engine rng{time(0)};

public:

    void add(Layer *layer, int pos=-1)
    {
        /*
            Pievieno slāni neironu tīklam pos pozīcijā
            Ja pos=-1, tad beigās
        */
        if(pos>=0)
        {
            assert(pos<=layers.size());
            auto it = layers.begin();
            advance(it, pos);
            layers.insert(it,layer);
        }
        else layers.push_back(layer);
        if(LayerDense *dense = dynamic_cast<LayerDense*>(layer)) trainableLayers.push_back(dense);
    }

    void remove(int pos)
    {
        /*
            Noņem slāni pozīcijā
        */
        assert(pos>0 && pos<layers.size());
        auto it = layers.begin();
        advance(it, pos);
        auto toRem = *it;
        layers.erase(it);
        trainableLayers.erase(find(trainableLayers.begin(), trainableLayers.end(), toRem));
        loadedLayers.erase(find(loadedLayers.begin(), loadedLayers.end(), toRem));
    }

    MatrixXd predict(MatrixXd x, bool training=false)
    {
        /*
            Aprēķina un atgriež neironu tīkla rezultātu no x ievades
            training ir false, ja tīkls netiek trenēts
        */
        auto y = x;
        for(Layer* layer : layers)
        {
            layer->forward(y, training);
            y = layer->getOutput();
        }
        return y;
    }

    void train(vector<MatrixXd>& x, vector<MatrixXd>& y, Loss *loss, Optimizer *optimizer, unsigned epochs=1, unsigned batchSize=1, double testPortion=0, bool shuffleRandom=false)
    {

    /*
        Trenē neironu tīklu ar x vērtībām, kurām jāatbilst y vērtībāi tajā pašā indeksā
        Piemēro loss zaudējumu un optimizer optimizāciju
        Epochs skaits ir cik reizes iziet cauri tiem pašiem datiem
        batchSize ir pēc cik liela datu daudzuma pielieto optimizāciju jeb izmaina neironu tīkla struktūru no iegūtajiem datiem
        testPortion ir procents(0 līdz 1), ko no x un y neizmanto tīkla trenešanai, bet zaudējuma aprēķināšanai, ko parādīt lietotājam.
        Daļas tiek izvēlētas jauktā secībā.
        shuffleRandom ir patiesumvērtība, kas nosaka, vai nepieciešams pēc katra epoch samaisīt trenēšanas masīvu
    */
        assert(batchSize>0 && batchSize<=x.size());
        assert(epochs>0);
        assert(x.size()==y.size());
        assert(testPortion>=0 && testPortion<1);

        //Sagatavo testu daļu un trenešanas daļu

        vector<int> indices;
        for(int i = 0; i<x.size(); i++)
        {
            indices.push_back(i);
        }
        shuffle(indices.begin(), indices.begin(), rng);

        vector<pair<MatrixXd*,MatrixXd*>> testSlice, trainSlice;
        int i;
        for(i = 0; i<x.size()*testPortion; i++)
        {
            testSlice.push_back(make_pair(&x[indices[i]], &y[indices[i]]));
        }
        for(; i<x.size(); i++)
        {
            trainSlice.push_back(make_pair(&x[indices[i]], &y[indices[i]]));
        }

        //Trenēšanas cikls

        for(auto epoch = 0; epoch<epochs; epoch++)
        {
            cout<<"Epoch: "<<epoch<<"    ";

            for(auto i = 0; i<trainSlice.size();)
            {
                auto batchPos = 0;
                for(; batchPos<batchSize && i<trainSlice.size(); batchPos++, i++)
                {
                    const auto& sampleX = *trainSlice[i].first,
                                sampleY = *trainSlice[i].second;
                    auto output = predict(sampleX, true);
                    auto dvalues = loss->backward(output, sampleY);
                    for(auto ri = layers.rbegin(); ri != layers.rend(); ri++)
                    {
                        (*ri)->backward(dvalues);
                        dvalues = (*ri)->getDinputs();
                    }
                }
                    optimizer->preUpdateParams();
                    for(auto dense : trainableLayers)
                    {
                        dense->calcGradients(batchPos+1);
                        optimizer->updateParams(*dense);
                    }
                    optimizer->postUpdateParams();

            }
            loss->newPass();
            for(const auto& [sampleX, sampleY] : testSlice)
            {
                auto output = predict(*sampleX);
                loss->calculate(output, *sampleY);
            }
            auto epochLoss = loss->calculateAccumulated();
            cout<<"Loss: "<<epochLoss<<endl;

            if(shuffleRandom)
            {
                shuffle(trainSlice.begin(), trainSlice.end(), rng);
            }

        }
    }

    void saveNetwork(const string& filename) const
    {
        /*
            Saglabā neironu tīklu failā, kas atrodas filename
        */

        /*
            Saglabāšans struktūra ([] - iekavas nebūs beigu rez.)
            [Tīkla identifaktors](parametrs,parametrs...)[papildus neobligāti dati];[Nākamais slānis];...

            piem. LD(1,1,0.0,0.4)w4b2;RE(0.2)

        */
        ofstream serialWriter(filename);

        string serial = "";
        for(auto layer : layers)
        {
            serial += layer->serialize() + ';';
        }
        //cout<<serial;
        serialWriter << serial;
        serialWriter.close();
    }

    void loadNetwork(const string& filename)
    {
        /*
            Ielādē šajā modelī filename atrodošo neironu tīklu, kas saglabāts ar šo programmu
        */
        ifstream serialFileReader(filename);
        if(!serialFileReader)
        {
            throw "Fails neeksitē";
        }
        string serial;
        serialFileReader >> serial;
        serialFileReader.close();
        int delimpos;
        while((delimpos = serial.find(';')) != string::npos) // sadala serial pa string, kas serial atdalīti ar ; jeb serial.split(';')
        {
            auto layerSerial = serial.substr(0,delimpos);
            auto layerType = layerSerial.substr(0,2);
            Layer *layerPtr;
            if(layerType == "LD")
            {
                layerPtr = LayerDense::deserialize(layerSerial);
            }
            else if(layerType == "DL")
            {
                layerPtr = LayerDropout::deserialize(layerSerial);
            }
            else if(layerType == "RE")
            {
                layerPtr = ActivationReLU::deserialize(layerSerial);
            }
            else if(layerType == "SI")
            {
                layerPtr = Sigmoid::deserialize(layerSerial);
            }
            add(layerPtr);
            loadedLayers.push_back(layerPtr);

            serial.erase(0, delimpos+1);
        }
    }

    ~Model()
    {
        //ielādētie slāņi no faila atrodas dinamiskajā atmiņā un ir jāizdzēš
        for(auto layer : loadedLayers)
        {
            delete layer;
        }
    }
};



int main()
{
   srand(time(0));
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1,50);


    vector<MatrixXd> x;
    vector<MatrixXd> y;

/*
    MatrixXd num(1,2), corr(1,1);
    for(int i = 0; i<15000; i++)
    {
        auto rand1 = dist(mt), rand2 = dist(mt);
        num.coeffRef(0,0) = rand1;
        num.coeffRef(0,1) = rand2;
        corr.coeffRef(0,0) = sqrt(rand1*rand1 + rand2*rand2);
        x.push_back(num);
        y.push_back(corr);
    }

   Model model;
   auto l1 = LayerDense(2,16, 0, 0.001);
   auto l2 = ActivationReLU(0.05);
   auto l3 = LayerDense(16,32, 0, 0.001);
   auto l4 = ActivationReLU(0.05);
   auto l5 = LayerDense(32,16, 0, 0.001);
   auto l6 = ActivationReLU(0.05);
   auto l7 = LayerDense(16,1);
   model.add(&l1);
   model.add(&l2);
   model.add(&l3);
   model.add(&l4);
   model.add(&l5);
   model.add(&l6);
   model.add(&l7);

*/
/*
    MatrixXd num(1,1), corr(1,1);
    for(int i = 0; i<15000; i++)
    {
        auto rand = dist(mt);
        num.coeffRef(0,0) = rand;
        corr.coeffRef(0,0) = rand*rand;
        x.push_back(num);
        y.push_back(corr);
    }

   Model model;
   auto l1 = LayerDense(1,16, 0, 0.001);
   auto l2 = ActivationReLU(0.05);
   auto l3 = LayerDense(16,16, 0, 0.001);
   auto l4 = ActivationReLU(0.05);
   auto l5 = LayerDense(16,1);
   model.add(&l1);
   model.add(&l2);
   model.add(&l3);
   model.add(&l4);
   model.add(&l5);
*/
/*

    MatrixXd num(1,1), corr(1,1);
    for(int i = 0; i<15000; i++)
    {
        auto rand = dist(mt);
        num.coeffRef(0,0) = rand;
        corr.coeffRef(0,0) = sin(rand);
        x.push_back(num);
        y.push_back(corr);
    }

   Model model;
   auto l1 = LayerDense(1,6, 0, 0.001);
   auto l2 = ActivationReLU(0.05);
   auto l3 = LayerDense(6,12, 0, 0.001);
   auto l4 = ActivationReLU(0.05);
   auto l5 = LayerDense(12,12, 0, 0.001);
   auto l6 = ActivationReLU(0.05);
   auto l7 = LayerDense(12,8, 0, 0.001);
   auto l8 = ActivationReLU(0.05);
   auto l9 = LayerDense(8,1);
   model.add(&l1);
   model.add(&l2);
   model.add(&l3);
   model.add(&l4);
   model.add(&l5);
   model.add(&l6);
   model.add(&l7);
   model.add(&l8);
   model.add(&l9);
*/
   /*
       MatrixXd num(1,1), corr(1,2);
    for(int i = 0; i<15000; i++)
    {
        auto rand = dist(mt);
        num.coeffRef(0,0) = rand;
        corr.coeffRef(0,0) = sqrt(rand);
        corr.coeffRef(0,1) = -sqrt(rand);
        x.push_back(num);
        y.push_back(corr);
    }

   Model model;
   auto l1 = LayerDense(1,12, 0, 0.001);
   auto l2 = ActivationReLU(0.05);
   auto l3 = LayerDense(12,8, 0, 0.001);
   auto l4 = ActivationReLU(0.05);
   auto l5 = LayerDense(8,32, 0, 0.001);
   auto l6 = ActivationReLU(0.05);
   auto l7 = LayerDense(32,2);
   model.add(&l1);
   model.add(&l2);
   model.add(&l3);
   model.add(&l4);
   model.add(&l5);
   model.add(&l6);
   model.add(&l7);
   */

   //Spēles papīrs-sķēres-akmentiņš rezultātu aproksimācija pēc principa
   //{s1 - akmens, s1 - sķēres, s1 - papīrs, s2 - akmens, s2 - sķēres, s2 - papīrs} => {s1 uzvar, s2 uzvar} (abi uzvar = neizšķirts)
   // for(int i = 0 ; i<5; i++)
   // {
   //     x.push_back((MatrixXd(1,6) << 0,0,1,0,0,1).finished());
   //     y.push_back((MatrixXd(1,2) << 1,1).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 0,0,1,0,1,0).finished());
   //     y.push_back((MatrixXd(1,2) << 0,1).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 0,1,0,0,0,1).finished());
   //     y.push_back((MatrixXd(1,2) << 1,0).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 0,1,0,0,1,0).finished());
   //     y.push_back((MatrixXd(1,2) << 1,1).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 0,1,0,1,0,0).finished());
   //     y.push_back((MatrixXd(1,2) << 0,1).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 0,0,1,0,0,1).finished());
   //     y.push_back((MatrixXd(1,2) << 1,1).finished());
   //
   //     x.push_back((MatrixXd(1,6) << 1,0,0,0,1,0).finished());
   //     y.push_back((MatrixXd(1,2) << 1,0).finished());
   //
   // }
   //
   // Model model;
   // auto l1 = LayerDense(6,32, 0, 0.001);
   // auto l2 = ActivationReLU(0.05);
   // auto l3 = LayerDense(32,32, 0, 0.001);
   // auto l4 = ActivationReLU(0.05);
   // auto l5 = LayerDense(32,2, 0, 0.001);
   // auto l6 = Sigmoid();
   //
   //
   // model.add(&l1);
   // model.add(&l2);
   // model.add(&l3);
   // model.add(&l4);
   // model.add(&l5);
   // model.add(&l6);
   //
   //
   // BinaryCrossEntropy bce;
   // Adam adam(0.001);
   //
   // model.train(x,y, (Loss*)&bce, (Optimizer*)&adam, 600, 1, 0.2);
   //
   // cout << model.predict((MatrixXd(1,6) << 0,0,1,0,1,0).finished()) <<endl; //01
   // cout << model.predict((MatrixXd(1,6) << 1,0,0,0,1,0).finished()) <<endl; //10
   // cout << model.predict((MatrixXd(1,6) << 0,0,1,0,0,1).finished()) <<endl; //11


/*
   Model modelSq, modelSqrt, modelSin, modelPyth;
   modelSq.loadNetwork("square.txt");
   modelSqrt.loadNetwork("sqrt.txt");
   modelSin.loadNetwork("sin.txt");
   modelPyth.loadNetwork("pyth.txt");

   MatrixXd test(1,1);
   test.coeffRef(0,0) = 20;
   cout<<modelSq.predict(test)<<endl;
   test.coeffRef(0,0) = -4;
   cout<<modelSq.predict(test)<<endl;
   test.coeffRef(0,0) = -12;
   cout<<modelSq.predict(test)<<endl;
   test.coeffRef(0,0) = 24; //576
   cout<<modelSq.predict(test)<<endl;
   test.coeffRef(0,0) = 33; //1089
   cout<<modelSq.predict(test)<<endl;
   test.coeffRef(0,0) = 57; //3249
   cout<<modelSq.predict(test)<<endl;
   cout<<endl;

   test.coeffRef(0,0) = 16;
   cout<<modelSqrt.predict(test)<<endl;
   test.coeffRef(0,0) = 25;
   cout<<modelSqrt.predict(test)<<endl;
   test.coeffRef(0,0) = 34;//5.8309
   cout<<modelSqrt.predict(test)<<endl;
   test.coeffRef(0,0) = 1;
   cout<<modelSqrt.predict(test)<<endl;
   test.coeffRef(0,0) = 2;//1.414
   cout<<modelSqrt.predict(test)<<endl;
   test.coeffRef(0,0) = 57;//5.8309
   cout<<modelSqrt.predict(test)<<endl;

   cout<<endl;


   test.coeffRef(0,0) = 0.7853; //sin(pi/4) = 0.70710678118
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = 4; //-0.7568
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = 13;//0.420167
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = 3.14;
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = 12;//-0.536572918
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = -40;//0.7451
   cout<<modelSin.predict(test)<<endl;
   test.coeffRef(0,0) = 57;//0.4361
   cout<<modelSin.predict(test)<<endl;

   cout<<endl;

   MatrixXd test2(1,2);
   test2.coeffRef(0,0) = 1;
   test2.coeffRef(0,1) = 1;
   cout<<modelPyth.predict(test2)<<endl; //1.414
   test2.coeffRef(0,0) = 6;
   test2.coeffRef(0,1) = 8;
   cout<<modelPyth.predict(test2)<<endl; //10
   test2.coeffRef(0,0) = 3;
   test2.coeffRef(0,1) = 15;
   cout<<modelPyth.predict(test2)<<endl; //15.297
   test2.coeffRef(0,0) = 34;
   test2.coeffRef(0,1) = 24;
   cout<<modelPyth.predict(test2)<<endl; //41.617
   test2.coeffRef(0,0) = 16;
   test2.coeffRef(0,1) = 42;
   cout<<modelPyth.predict(test2)<<endl; //44.944
   test2.coeffRef(0,0) = 48;
   test2.coeffRef(0,1) = 62;
   cout<<modelPyth.predict(test2)<<endl; //78.409
    */
  // model.saveNetwork("pyth.txt");
}
