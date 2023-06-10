pipeline {
  agent {
    label "ubuntu-latest"
  }
  
  stages {
    stage("Checkout repository") {
      steps {
        checkout scm
      }
    }
    
    stage("Set up Python") {
      steps {
        script {
          def pythonVersion = "3.8"
          tool "Python $pythonVersion"
        }
      }
    }
    
    stage("Install dependencies") {
      steps {
        sh "python -m pip install --upgrade pip"
        sh "pip install -r requirements.txt"
      }
    }
    
    stage("Set up DVC") {
      steps {
        sh "pip install dvc"
        sh "pip install dvc-gdrive"
      }
    }
    
    stage("Configure Google Drive remote") {
      steps {
        sh "dvc remote add -f gdrive gdrive:1TcslBQ2BQ5kNbIYLbQENGpW1UfrR1rH-"
      }
    }
    
    stage("Pull DVC-tracked data") {
      steps {
        sh "dvc pull -r gdrive"
      }
    }
    
    stage("Load DVC datasets") {
      steps {
        sh "dvc import route_github.csv.dvc Data_Files/"
      }
    }
    stage("Train models") {
      steps {
        sh "python MLFlow_route.py"
      }
    }
    
    stage("Run training") {
      steps {
        sh "python vs_anomaly.py --data ../Data_Files/route_github.csv"
      }
    }
    
    stage("Save models to DVC") {
      steps {
        dir("..") {
          sh "dvc add model_route.pkl"
          sh "dvc push"
        }
      }
    }
    
    stage("Deploy Flask app") {
      environment {
        MODEL1_PATH = "models/model_route.pkl"
      }
      steps {
        sh "python server.py"
      }
    }
  }
}
