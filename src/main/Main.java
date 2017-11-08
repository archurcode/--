package main;

import java.io.File;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Main {

    public static void main(String[] args) {
        Instances ins_train = null;
        Instances ins_test = null;
        Classifier cfs = null;
        try {
            File file_train = new File("rule_1_1.arff");
            ArffLoader loader_train = new ArffLoader();
            loader_train.setFile(file_train);
            
            File file_test = new File("rule_1_test.arff");
            ArffLoader loader_test = new ArffLoader();
            loader_test.setFile(file_test);
            // 70
            ins_train = loader_train.getDataSet();
            // 
            ins_test = loader_test.getDataSet();
            
            ins_train.setClassIndex(ins_train.numAttributes() - 1);
            ins_test.setClassIndex(ins_train.numAttributes() - 1);
            
            cfs = new JRip();
            
            int numOftest = 4500;
            final int t = 30;
            
            int loops = 100;
            int interval = t;
            int times = t;
            double f = 1.0;
            double accuracy = 0.0;
            double threshold = 0.95;
            int same = 0;
            int index = 0;
            Instance testInst;
            
            DecimalFormat df1 = new DecimalFormat("#0.000000");
            DecimalFormat df2 = new DecimalFormat("#0.000");
            
            while(loops>0 && index<numOftest){
                
                for (int i = 0; i < interval; i++) {
                    if(times>0){
                        cfs.buildClassifier(ins_train);
                        
                        testInst = ins_test.instance(index);
                        double predictValue = cfs.classifyInstance(testInst);
//                        System.out.println(i+" "+testInst.classValue()+"--"+predictValue);
                        if(testInst.classValue() == predictValue){
                            same++;
                        }
                        ins_train.add(ins_test.instance(index));
                        
                        index++;
                        times--;
                    }else{
                        break;
                    }
                }
                
                times = t;
                accuracy = same/(times*1.0);
//                
//                if(accuracy >= threshold){
//                    f = f/2;
//                    interval = (int)Math.ceil(times/f); //向上取整
//                }else{
//                    f = Math.min((f+1)/10.0, 1.0);
//                    interval = (int)Math.floor(times/f); //向下取整
//                }
                
                if(accuracy >= threshold){
                    interval += 10;
                }else{
//                    interval = Math.max(interval/2, t); 
                    interval = t + loops;
                }
                
                f = t/(interval*1.0);
                
                String tmp_freq = df1.format(f);
                String tmp_acc = df2.format(accuracy);
                System.out.println(tmp_freq +" "+ tmp_acc +" "+ interval);
//                System.out.println(loops +": "+ tmp_freq +" "+ tmp_acc +" "+ interval);
//                System.out.println("f = " + tmp + " acc = " + accuracy);
                
                
                same = 0;
                loops--;
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}

























//cfs.buildClassifier(ins_train);
//
//Instance testInst;
//Evaluation testingEvaluation = new Evaluation(ins_train);
//int length = ins_train.numInstances();
//
//for (int i = 0; i < length; i++) {
//  testInst = ins_train.instance(i);
//  double predictValue = testingEvaluation.evaluateModelOnceAndRecordPrediction(cfs,
//          testInst);
//  System.out.println(i+" "+testInst.classValue()+"--"+predictValue);
//}