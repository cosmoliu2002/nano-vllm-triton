import sys
import yaml

def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_config.py <config_file> <key>", file=sys.stderr)
        sys.exit(1)
    
    config_file = sys.argv[1]
    key = sys.argv[2]
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if key in config:
            print(config[key])
        else:
            print(f"Key '{key}' not found in config", file=sys.stderr)
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Config file '{config_file}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()