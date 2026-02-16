-- Create royalty_calculations table
CREATE TABLE public.royalty_calculations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  royalty_statement_id UUID REFERENCES public.project_files(id) ON DELETE CASCADE NOT NULL,
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  user_id UUID NOT NULL, -- Assuming user_id is managed by Auth, usually references auth.users but keeping as UUID for flexibility if not strictly linked in public schema
  results JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS
ALTER TABLE public.royalty_calculations ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own calculations"
ON public.royalty_calculations
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own calculations"
ON public.royalty_calculations
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own calculations"
ON public.royalty_calculations
FOR DELETE
USING (auth.uid() = user_id);

-- Create trigger for updated_at
CREATE TRIGGER update_royalty_calculations_updated_at
BEFORE UPDATE ON public.royalty_calculations
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();


-- Create junction table for contracts
CREATE TABLE public.royalty_calculation_contracts (
  calculation_id UUID REFERENCES public.royalty_calculations(id) ON DELETE CASCADE NOT NULL,
  contract_id UUID REFERENCES public.project_files(id) ON DELETE CASCADE NOT NULL,
  PRIMARY KEY (calculation_id, contract_id)
);

-- Enable RLS
ALTER TABLE public.royalty_calculation_contracts ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own calculation contracts"
ON public.royalty_calculation_contracts
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.royalty_calculations rc
    WHERE rc.id = royalty_calculation_contracts.calculation_id
    AND rc.user_id = auth.uid()
  )
);

CREATE POLICY "Users can insert their own calculation contracts"
ON public.royalty_calculation_contracts
FOR INSERT
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.royalty_calculations rc
    WHERE rc.id = royalty_calculation_contracts.calculation_id
    AND rc.user_id = auth.uid()
  )
);

CREATE POLICY "Users can delete their own calculation contracts"
ON public.royalty_calculation_contracts
FOR DELETE
USING (
  EXISTS (
    SELECT 1 FROM public.royalty_calculations rc
    WHERE rc.id = royalty_calculation_contracts.calculation_id
    AND rc.user_id = auth.uid()
  )
);


-- Create function and trigger to delete calculation if a contract is deleted
-- (The junction row is deleted by CASCADE, this trigger cleans up the parent calculation)
CREATE OR REPLACE FUNCTION public.delete_orphan_calculation()
RETURNS TRIGGER AS $$
BEGIN
  -- Delete the parent calculation. 
  -- This will cascade delete other junction rows for this calculation, 
  -- which will re-trigger this function, but the DELETE on the parent will be a no-op for subsequent calls.
  DELETE FROM public.royalty_calculations WHERE id = OLD.calculation_id;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER cleanup_calculation_on_contract_delete
AFTER DELETE ON public.royalty_calculation_contracts
FOR EACH ROW
EXECUTE FUNCTION public.delete_orphan_calculation();
