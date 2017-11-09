package main;

import java.io.File;
import java.text.DecimalFormat;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Main {

    public static void main(String[] args) {
        Instances ins_train = null;
        Instances ins_test = null;
        Classifier cfs = null;
        
        DecimalFormat df1 = new DecimalFormat("#0.000000");
        DecimalFormat df2 = new DecimalFormat("#0.000");
        
        try {
            File file_train = new File("rule_1_1.arff");
            ArffLoader loader_train = new ArffLoader();
            loader_train.setFile(file_train);
            
            File file_test = new File("1-4_rule_data.arff");
            ArffLoader loader_test = new ArffLoader();
            loader_test.setFile(file_test);
            
            ins_train = loader_train.getDataSet();
            ins_test = loader_test.getDataSet();
            
            ins_train.setClassIndex(ins_train.numAttributes() - 1);
            ins_test.setClassIndex(ins_train.numAttributes() - 1);
            
            cfs = new JRip();
            
            int numOftest = 1000;
            final int init = 25;
            
            //parameter
            int interval = init;
            int times = init;
            
            //target
            double freq = 1.0;
            double accuracy = 0.0;
            double threshold = 0.70;
            
            int same = 0;
            int index = 0;
            
            Instance testInst;
            
            LinkedList<String> window = new LinkedList<String>();
            
            int step = 1;
            
            while(numOftest>0 && index<=6000) {
                
                //build classifier
                cfs.buildClassifier(ins_train); 
                //get one test instance
                testInst = ins_test.instance(index);
                //get predict value
                double predictValue = cfs.classifyInstance(testInst);
                
                ins_train.add(testInst);
                
                if(times>0) {
                    //no accuracy 
                    if(testInst.classValue() == predictValue){
                        window.addLast("T");
                        same++;
                    }else{
                        window.addLast("F");
                    }
                    
                    times--;
//                    System.out.println("same "+same);
                    accuracy = same/((init-times)*1.0);
                }else {
                    if(testInst.classValue() == predictValue){
                        window.addLast("T");
                        same++;
                    }else{
                        window.addLast("F");
                    }
                    
                    if("T".equals(window.pollFirst())){
                        same--;
                    }
                    
                    //compute accuracy
//                    System.out.println("same = "+same);
                    accuracy = same/(init*1.0);
                    
                    
                }
                
                if(accuracy >= threshold){
                    interval += 5;
                }else{
                    interval = init;
                }
                
                freq = init/(interval*1.0);
                
                step = (int) Math.ceil( interval/(init*1.0) );
                
                String tmp_freq = df1.format(freq);
                String tmp_acc = df2.format(accuracy);
              
                System.out.println(tmp_freq +" "+ tmp_acc +" "+ interval);
                
                index = index + step;
//                System.out.println("index = "+index);
                
                numOftest--;
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}





//
//String tmp_freq = df1.format(freq);
//String tmp_acc = df2.format(accuracy);
//
//System.out.println(tmp_freq +" "+ tmp_acc +" "+ interval);
