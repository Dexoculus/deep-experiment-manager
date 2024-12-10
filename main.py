from experiment_manager import ExperimentManager

def main():
    configs = './test/config.yaml'
    manager = ExperimentManager(config_path=configs)

    manager.run_experiment()

if __name__ == '__main__':
    main()