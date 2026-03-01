import argparse

def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="A tool to calculate hospital capacity.", 
                                     allow_abbrev=True)

    # 2. Add arguments
    # Positional argument (Required)
    parser.add_argument("hospital_name", type=str, help="The name of the facility.")
    
    # Optional argument with a flag (Default value)
    parser.add_argument("--beds", choices=[3,5],
                        type=int, default=10, help="Number of available beds.")
    
    # A 'Switch' or Boolean flag
    parser.add_argument("--emergency", action="store_true", help="Set if in emergency mode.")

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Use the arguments
    print(f"Hospital: {args.hospital_name}")
    print(f"Capacity: {args.beds} beds")
    
    if args.emergency:
        print("STATUS: Emergency Mode Active!")
    else:
        print("STATUS: Normal Operations.")

if __name__ == "__main__":
    main()